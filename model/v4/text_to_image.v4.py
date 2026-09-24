"""
Fine-tuning version of the 60x60 conditional text-to-image GAN.

This version starts from the completed 300-epoch checkpoint and:
- keeps the generator LR at 2e-4,
- reduces discriminator LR to 1e-4,
- keeps the existing architecture/losses,
- monitors nine different captions with fixed noise,
- saves fine-tuned models separately.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


IMG_SIZE = 60
CHANNELS = 3
BATCH_SIZE = 16
NOISE_DIM = 128
# V4 starts from the saved V2 epoch-500 models.
FINE_TUNE_EPOCHS = 150
START_EPOCH = 500

MAX_LEN = 20
VOCAB_SIZE = 5000
EMBED_DIM = 128
DISC_TEXT_DIM = 128
N_CRITIC = 1

GEN_LR = 2e-4
DISC_LR = 1e-4

# Controlled V4 change: strengthen real-image / wrong-caption discrimination.
MISMATCH_LOSS_WEIGHT = 0.75

EMA_DECAY = 0.995
EMA_UPDATE_EVERY = 5
MONITOR_GRID = 9
LOG_EVERY_STEPS = 20

V2_MODEL_DIR = "./model/v2"
V4_MODEL_DIR = "./model/v4"
V4_CHECKPOINT_DIR = os.path.join(V4_MODEL_DIR, "checkpoints")

V2_GENERATOR_PATH = os.path.join(V2_MODEL_DIR, "generator_model_60px_finetuned.keras")
V2_EMA_GENERATOR_PATH = os.path.join(
    V2_MODEL_DIR, "generator_ema_model_60px_finetuned.keras"
)
V2_TEXT_ENCODER_PATH = os.path.join(V2_MODEL_DIR, "text_encoder_60px_finetuned.keras")
V2_DISCRIMINATOR_PATH = os.path.join(
    V2_MODEL_DIR, "discriminator_model_60px_finetuned.keras"
)
V2_TOKENIZER_PATH = os.path.join(V2_MODEL_DIR, "tokenizer_60px_finetuned.json")

V4_GENERATOR_PATH = os.path.join(V4_MODEL_DIR, "generator_model_60px_v4.keras")
V4_EMA_GENERATOR_PATH = os.path.join(V4_MODEL_DIR, "generator_ema_model_60px_v4.keras")
V4_TEXT_ENCODER_PATH = os.path.join(V4_MODEL_DIR, "text_encoder_60px_v4.keras")
V4_DISCRIMINATOR_PATH = os.path.join(V4_MODEL_DIR, "discriminator_model_60px_v4.keras")

MONITOR_CAPTIONS = [
    "a body of water",
    "a mountain landscape",
    "a green field",
    "a sunset over the ocean",
    "a city skyline",
    "a forest with trees",
    "a road through the mountains",
    "a beach with waves",
    "a lake with mountains in the background",
]
USE_MIXED_PRECISION = False
USE_XLA = False
CPU_INTRA_OP_THREADS = 4
CPU_INTER_OP_THREADS = 2
AUTOTUNE = tf.data.AUTOTUNE


def configure_runtime():
    try:
        tf.config.threading.set_intra_op_parallelism_threads(CPU_INTRA_OP_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(CPU_INTER_OP_THREADS)
    except RuntimeError:
        pass

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if USE_XLA:
        tf.config.optimizer.set_jit(True)

    if USE_MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")


def make_text_encoder():
    text_input = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name="token_ids")

    x = layers.Embedding(
        input_dim=VOCAB_SIZE + 1,
        output_dim=EMBED_DIM,
        mask_zero=True,
        name="text_embedding",
    )(text_input)

    x = layers.LSTM(EMBED_DIM, name="text_lstm")(x)
    x = layers.LayerNormalization(name="text_norm")(x)

    return tf.keras.Model(text_input, x, name="text_encoder")


class ConditioningAugmentation(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_mean = layers.Dense(embed_dim)
        self.dense_log_sigma = layers.Dense(embed_dim)

    def call(self, inputs):
        mean = self.dense_mean(inputs)
        log_sigma = self.dense_log_sigma(inputs)
        log_sigma = tf.clip_by_value(log_sigma, -4.0, 4.0)
        stddev = tf.exp(0.5 * log_sigma)
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + stddev * epsilon


def _decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=CHANNELS)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img


def _augment_image(img, text):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.08)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.clip_by_value(img, -1.0, 1.0)
    return img, text


def load_image_caption_dataset(img_folder, caption_file):
    image_paths = []
    captions = []

    with open(caption_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line or "|" not in line:
                continue

            img_name, caption = line.split("|", 1)

            image_paths.append(os.path.join(img_folder, img_name))

            captions.append(caption)

    sequences = tokenizer.texts_to_sequences(captions)

    padded = pad_sequences(sequences, maxlen=MAX_LEN).astype(np.int32)

    image_ds = tf.data.Dataset.from_tensor_slices(image_paths).map(
        _decode_image, num_parallel_calls=AUTOTUNE
    )

    text_ds = tf.data.Dataset.from_tensor_slices(padded)

    dataset = tf.data.Dataset.zip((image_ds, text_ds))

    return (
        dataset.cache()
        .shuffle(min(1000, len(image_paths)), reshuffle_each_iteration=True)
        .map(_augment_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )


def make_generator():
    noise_input = tf.keras.Input(shape=(NOISE_DIM,), name="noise_input")
    text_input = tf.keras.Input(shape=(EMBED_DIM,), name="text_input")
    ca = ConditioningAugmentation(EMBED_DIM)(text_input)

    x = layers.Concatenate()([noise_input, ca])
    x = layers.Dense(5 * 5 * 128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Reshape((5, 5, 128))(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(16, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Resizing(IMG_SIZE, IMG_SIZE)(x)
    x = layers.Conv2D(8, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    output = layers.Conv2D(CHANNELS, kernel_size=3, padding="same", activation="tanh")(
        x
    )

    return tf.keras.Model([noise_input, text_input], output, name="generator")


def make_discriminator():
    image_input = tf.keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, CHANNELS), name="image_input"
    )
    text_input = tf.keras.Input(shape=(EMBED_DIM,), name="text_input")

    x = layers.Conv2D(32, 4, strides=2, padding="same")(image_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 4, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)

    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)

    image_features = layers.Dense(EMBED_DIM, name="image_projection")(x)
    text_features = layers.Dense(EMBED_DIM, name="text_projection")(text_input)
    compatibility = layers.Dot(axes=1, normalize=True, name="image_text_compatibility")(
        [image_features, text_features]
    )
    realness = layers.Dense(1, name="realness")(x)

    output = layers.Add(name="conditional_score")([realness, compatibility])

    return tf.keras.Model(
        [image_input, text_input], output, name="conditional_discriminator"
    )


def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(tf.nn.relu(1.0 - real_output)) + tf.reduce_mean(
        tf.nn.relu(1.0 + fake_output)
    )


@tf.function(reduce_retracing=True)
def disc_step(
    images,
    caption_tokens,
    text_encoder,
    generator,
    discriminator,
    disc_opt,
):
    batch_size = tf.shape(images)[0]

    with tf.GradientTape() as tape:
        text_features = text_encoder(caption_tokens, training=True)
        noise = tf.random.normal([batch_size, NOISE_DIM])
        fake_images = generator([noise, text_features], training=True)
        fake_images_for_d = tf.stop_gradient(fake_images)

        real_output = discriminator([images, text_features], training=True)
        fake_output = discriminator([fake_images_for_d, text_features], training=True)

        mismatched_tokens = make_mismatched_captions(caption_tokens)
        mismatched_features = text_encoder(mismatched_tokens, training=True)
        mismatch_output = discriminator([images, mismatched_features], training=True)

        d_loss = discriminator_loss(real_output, fake_output)
        mismatch_loss = tf.reduce_mean(tf.nn.relu(1.0 + mismatch_output))

        d_loss = d_loss + MISMATCH_LOSS_WEIGHT * mismatch_loss

    trainable_vars = (
        discriminator.trainable_variables + text_encoder.trainable_variables
    )
    grads = tape.gradient(d_loss, trainable_vars)
    disc_opt.apply_gradients(zip(grads, trainable_vars))

    return d_loss


@tf.function(reduce_retracing=True)
def gen_step(
    caption_tokens,
    text_encoder,
    generator,
    discriminator,
    gen_opt,
):
    batch_size = tf.shape(caption_tokens)[0]

    with tf.GradientTape() as tape:
        text_features = text_encoder(
            caption_tokens,
            training=True,
        )

        noise = tf.random.normal([batch_size, NOISE_DIM])

        fake_images = generator(
            [noise, text_features],
            training=True,
        )

        fake_output = discriminator(
            [fake_images, text_features],
            training=True,
        )

        g_loss = generator_loss(
            fake_output,
        )

    trainable_vars = generator.trainable_variables + text_encoder.trainable_variables

    grads = tape.gradient(
        g_loss,
        trainable_vars,
    )

    gen_opt.apply_gradients(zip(grads, trainable_vars))

    return g_loss


def make_mismatched_captions(caption_tokens):
    """Create negative text/image pairs by rotating captions within a batch."""
    batch_size = tf.shape(caption_tokens)[0]

    def rotate():
        return tf.roll(caption_tokens, shift=1, axis=0)

    return tf.cond(
        tf.greater(batch_size, 1),
        rotate,
        lambda: caption_tokens,
    )


def build_ema_generator(generator):
    ema = make_generator()
    ema.set_weights(generator.get_weights())
    ema.trainable = False
    return ema


@tf.function
def update_ema(model, ema_model, decay):
    tf.debugging.assert_equal(
        len(model.weights),
        len(ema_model.weights),
    )

    for v_ema, v in zip(ema_model.weights, model.weights):
        v_ema.assign(decay * v_ema + (1.0 - decay) * v)


def save_generated_samples(
    epoch_number, folder_type, generator, monitor_noise, monitor_text
):
    images = generator([monitor_noise, monitor_text], training=False)
    images = tf.clip_by_value((images + 1.0) / 2.0, 0.0, 1.0)

    n = int(images.shape[0])
    grid = int(np.ceil(np.sqrt(n)))
    pad = grid * grid - n
    if pad > 0:
        images = tf.concat(
            [images, tf.zeros([pad, IMG_SIZE, IMG_SIZE, CHANNELS], dtype=images.dtype)],
            axis=0,
        )
    tiled = tf.reshape(images, [grid, grid, IMG_SIZE, IMG_SIZE, CHANNELS])
    tiled = tf.transpose(tiled, [0, 2, 1, 3, 4])
    tiled = tf.reshape(tiled, [grid * IMG_SIZE, grid * IMG_SIZE, CHANNELS])
    tiled_u8 = tf.cast(tiled * 255.0, tf.uint8)

    out_dir = os.path.join(V4_MODEL_DIR, "gen_images", folder_type)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"generated_image_epoch_{epoch_number}.png")
    tf.io.write_file(out_path, tf.io.encode_png(tiled_u8))


def _resolve_captions_path() -> str:
    v2 = "./data/captions_60px_v2.txt"
    v1 = "./data/captions.txt"
    return v2 if os.path.exists(v2) else v1


def main():
    configure_runtime()
    os.makedirs(V4_MODEL_DIR, exist_ok=True)
    os.makedirs(V4_CHECKPOINT_DIR, exist_ok=True)

    captions_path = _resolve_captions_path()
    print(f"Using captions file: {captions_path}")

    if not os.path.isfile(V2_TOKENIZER_PATH):
        raise FileNotFoundError(f"V2 tokenizer not found: {V2_TOKENIZER_PATH}")

    # IMPORTANT: use the exact V2 tokenizer mapping. Do not refit it.
    with open(V2_TOKENIZER_PATH, "r", encoding="utf-8") as f:
        global tokenizer
        tokenizer = tokenizer_from_json(f.read())

    payload = load_image_caption_dataset("./data/image60px", captions_path)

    required = [
        V2_GENERATOR_PATH,
        V2_EMA_GENERATOR_PATH,
        V2_TEXT_ENCODER_PATH,
        V2_DISCRIMINATOR_PATH,
    ]

    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Missing V2 model files:\n" + "\n".join(f"  - {path}" for path in missing)
        )

    print("Loading V2 epoch-500 models...")
    generator = tf.keras.models.load_model(
        V2_GENERATOR_PATH,
        custom_objects={"ConditioningAugmentation": ConditioningAugmentation},
        compile=False,
    )

    g_ema = tf.keras.models.load_model(
        V2_EMA_GENERATOR_PATH,
        custom_objects={"ConditioningAugmentation": ConditioningAugmentation},
        compile=False,
    )

    text_encoder = tf.keras.models.load_model(
        V2_TEXT_ENCODER_PATH,
        compile=False,
    )

    discriminator = tf.keras.models.load_model(
        V2_DISCRIMINATOR_PATH,
        compile=False,
    )

    print(f"Text encoder parameters: {text_encoder.count_params():,}")
    print(f"Generator parameters: {generator.count_params():,}")
    print(f"Discriminator parameters: {discriminator.count_params():,}")
    print(f"V4 mismatch loss weight: {MISMATCH_LOSS_WEIGHT}")

    gen_opt = tf.keras.optimizers.Adam(
        learning_rate=GEN_LR,
        beta_1=0.5,
        beta_2=0.999,
    )

    disc_opt = tf.keras.optimizers.Adam(
        learning_rate=DISC_LR,
        beta_1=0.5,
        beta_2=0.999,
    )

    target_epoch = START_EPOCH + FINE_TUNE_EPOCHS
    epoch_var = tf.Variable(
        START_EPOCH,
        dtype=tf.int64,
        trainable=False,
        name="v4_epoch",
    )

    checkpoint_prefix = os.path.join(
        V4_CHECKPOINT_DIR,
        "ckpt_60px_conditional_v4",
    )

    checkpoint = tf.train.Checkpoint(
        epoch=epoch_var,
        text_encoder=text_encoder,
        generator=generator,
        discriminator=discriminator,
        g_ema=g_ema,
        g_optimizer=gen_opt,
        d_optimizer=disc_opt,
    )

    latest_ckpt = tf.train.latest_checkpoint(V4_CHECKPOINT_DIR)

    if latest_ckpt:
        checkpoint.restore(latest_ckpt).expect_partial()
        print(f"Resumed V4 checkpoint: {latest_ckpt}")
    else:
        print("Starting V4 from saved V2 epoch-500 models.")

    gen_opt.learning_rate.assign(GEN_LR)
    disc_opt.learning_rate.assign(DISC_LR)

    start_epoch = int(epoch_var.numpy())

    print(
        f"Training schedule: epochs {start_epoch + 1}-{target_epoch} | "
        f"gen_lr={float(gen_opt.learning_rate):.1e} | "
        f"disc_lr={float(disc_opt.learning_rate):.1e} | "
        f"mismatch_weight={MISMATCH_LOSS_WEIGHT:.2f}"
    )

    # Fixed noise + different captions: monitor caption conditioning.
    monitor_noise = tf.random.stateless_normal(
        [MONITOR_GRID, NOISE_DIM],
        seed=[SEED, 0],
    )

    seq = tokenizer.texts_to_sequences(MONITOR_CAPTIONS)
    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="pre",
        truncating="pre",
    ).astype(np.int32)

    monitor_tokens = tf.constant(padded, dtype=tf.int32)
    monitor_text = text_encoder(monitor_tokens, training=False)

    g_loss_metric = tf.keras.metrics.Mean()
    d_loss_metric = tf.keras.metrics.Mean()

    while int(epoch_var.numpy()) < target_epoch:
        g_loss_metric.reset_state()
        d_loss_metric.reset_state()
        epoch_start = time.time()

        actual_epoch = int(epoch_var.numpy()) + 1

        for step_idx, (image_batch, caption_batch) in enumerate(payload):
            d_loss = disc_step(
                image_batch,
                caption_batch,
                text_encoder,
                generator,
                discriminator,
                disc_opt,
            )
            d_loss_metric.update_state(d_loss)

            if step_idx % N_CRITIC == 0:
                g_loss = gen_step(
                    caption_batch,
                    text_encoder,
                    generator,
                    discriminator,
                    gen_opt,
                )
                g_loss_metric.update_state(g_loss)

                if step_idx % EMA_UPDATE_EVERY == 0:
                    update_ema(generator, g_ema, EMA_DECAY)

            if (step_idx + 1) % LOG_EVERY_STEPS == 0:
                sys.stdout.write(
                    f"\rEpoch {actual_epoch:>4}  step {step_idx+1:>5}  "
                    f"g={float(g_loss_metric.result()):.4f}  "
                    f"d={float(d_loss_metric.result()):.4f}"
                )
                sys.stdout.flush()

        elapsed = time.time() - epoch_start
        g_avg = float(g_loss_metric.result())
        d_avg = float(d_loss_metric.result())

        epoch_var.assign(actual_epoch)

        sys.stdout.write("\r" + " " * 80 + "\r")
        print(
            f"Epoch {actual_epoch}/{target_epoch}  "
            f"Gen {g_avg:.4f}  Disc {d_avg:.4f}  ({elapsed:.1f}s)"
        )

        if actual_epoch % 10 == 0 or actual_epoch == target_epoch:
            save_generated_samples(
                actual_epoch,
                "v4",
                g_ema,
                monitor_noise,
                monitor_text,
            )

        if actual_epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    generator.save(V4_GENERATOR_PATH)
    g_ema.save(V4_EMA_GENERATOR_PATH)
    discriminator.save(V4_DISCRIMINATOR_PATH)
    text_encoder.save(V4_TEXT_ENCODER_PATH)

    with open(
        os.path.join(V4_MODEL_DIR, "tokenizer_60px_v4.json"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(tokenizer.to_json())

    print()
    print("V4 training complete.")
    print(f"Models saved in: {V4_MODEL_DIR}")


if __name__ == "__main__":
    main()
