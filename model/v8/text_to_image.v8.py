"""V8 — Clean-caption conditional GAN baseline.

V8 changes direction from repeated generator-architecture experiments to
training-data/supervision quality.

Important:
- V8 reads the SAME data/captions.txt path.
- The original caption file should be backed up before replacing it.
- V8 builds a NEW tokenizer and NEW text encoder from the current captions.
- V8 starts from scratch intentionally; old V2-V7 text embeddings are not
  reused because the caption/token distribution may have changed.
- The generator uses the stable V4-style architecture rather than V6/V7
  multi-stage conditioning.
- The discriminator keeps projection conditioning and wrong-caption training.
- No V5 alignment loss, V6 additive conditioning, or V7 FiLM is used.

This makes V8 a clean experiment:
better/cleaner captions + stable GAN architecture.
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
import re
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = 60
CHANNELS = 3
BATCH_SIZE = 16
NOISE_DIM = 128

EPOCHS = 300
MAX_LEN = 20
VOCAB_SIZE = 5000
EMBED_DIM = 128
N_CRITIC = 1

GEN_LR = 2e-4
DISC_LR = 1e-4

MISMATCH_LOSS_WEIGHT = 0.75

EMA_DECAY = 0.995
EMA_UPDATE_EVERY = 5

MONITOR_GRID = 9
LOG_EVERY_STEPS = 20

USE_MIXED_PRECISION = False
USE_XLA = False
CPU_INTRA_OP_THREADS = 4
CPU_INTER_OP_THREADS = 2
AUTOTUNE = tf.data.AUTOTUNE

CAPTION_FILE = "./data/captions.txt"
IMAGE_DIR = "./data/image60px"

V8_MODEL_DIR = "./model/v8"
V8_CHECKPOINT_DIR = os.path.join(V8_MODEL_DIR, "checkpoints")
V8_DATA_DIR = os.path.join(V8_MODEL_DIR, "data")

GENERATOR_PATH = os.path.join(V8_MODEL_DIR, "generator_model_60px_v8.keras")
EMA_GENERATOR_PATH = os.path.join(V8_MODEL_DIR, "generator_ema_model_60px_v8.keras")
DISCRIMINATOR_PATH = os.path.join(V8_MODEL_DIR, "discriminator_model_60px_v8.keras")
TEXT_ENCODER_PATH = os.path.join(V8_MODEL_DIR, "text_encoder_60px_v8.keras")
TOKENIZER_PATH = os.path.join(V8_MODEL_DIR, "tokenizer_60px_v8.json")
CLEAN_CAPTIONS_PATH = os.path.join(V8_DATA_DIR, "captions_used_v8.txt")

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


def configure_runtime():
    try:
        tf.config.threading.set_intra_op_parallelism_threads(CPU_INTRA_OP_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(CPU_INTER_OP_THREADS)
    except RuntimeError:
        pass

    if USE_XLA:
        tf.config.optimizer.set_jit(True)

    if USE_MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


class ConditioningAugmentation(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_mean = layers.Dense(embed_dim, name="ca_mean")
        self.dense_log_sigma = layers.Dense(embed_dim, name="ca_log_sigma")

    def call(self, inputs):
        mean = self.dense_mean(inputs)
        log_sigma = self.dense_log_sigma(inputs)
        log_sigma = tf.clip_by_value(log_sigma, -4.0, 4.0)
        stddev = tf.exp(0.5 * log_sigma)
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + stddev * epsilon

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


def clean_caption(caption):
    """
    Conservative caption normalization.

    We intentionally do NOT invent visual content. The cleaner only fixes
    obvious textual artifacts:
      - whitespace
      - repeated adjacent words
      - repeated punctuation
      - accidental leading/trailing punctuation

    A caption is rejected only when it is empty or clearly unusable.
    """
    caption = caption.strip().lower()
    caption = re.sub(r"\s+", " ", caption)
    caption = re.sub(r"([,.!?])\1+", r"\1", caption)

    words = caption.split()
    cleaned_words = []

    for word in words:
        normalized = re.sub(r"[^\w']+", "", word)
        if (
            cleaned_words
            and normalized
            and re.sub(r"[^\w']+", "", cleaned_words[-1]) == normalized
        ):
            continue
        cleaned_words.append(word)

    caption = " ".join(cleaned_words)
    caption = caption.strip(" ,.!?")

    if not caption:
        return None

    # BLIP occasionally produces an unusable single-token artifact.
    if caption in {"thumb", "thumbnail"}:
        return None

    if len(caption.split()) < 2:
        return None

    return caption


def prepare_v8_captions():
    if not os.path.isfile(CAPTION_FILE):
        raise FileNotFoundError(f"Caption file not found: {CAPTION_FILE}")

    os.makedirs(V8_DATA_DIR, exist_ok=True)

    rows = []
    rejected = 0
    changed = 0
    total = 0

    with open(CAPTION_FILE, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or "|" not in line:
                continue

            image_name, raw_caption = line.split("|", 1)
            image_name = image_name.strip()
            raw_caption = raw_caption.strip()

            total += 1

            cleaned = clean_caption(raw_caption)

            if cleaned is None:
                rejected += 1
                continue

            if cleaned != raw_caption.lower():
                changed += 1

            image_path = os.path.join(IMAGE_DIR, image_name)

            if not os.path.isfile(image_path):
                rejected += 1
                continue

            rows.append((image_name, cleaned))

    if len(rows) < BATCH_SIZE:
        raise RuntimeError(f"Only {len(rows)} usable image-caption pairs remain.")

    with open(
        CLEAN_CAPTIONS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        for image_name, caption in rows:
            f.write(f"{image_name}|{caption}\n")

    print()
    print("V8 caption preparation")
    print("=======================")
    print(f"Original pairs:       {total}")
    print(f"Usable pairs:         {len(rows)}")
    print(f"Rejected pairs:       {rejected}")
    print(f"Normalized captions:  {changed}")
    print(f"Used caption file:    {CLEAN_CAPTIONS_PATH}")
    print()

    return rows


def build_tokenizer(rows):
    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token="<unk>",
        lower=True,
    )

    captions = [caption for _, caption in rows]
    tokenizer.fit_on_texts(captions)

    print(
        f"Tokenizer vocabulary: {min(len(tokenizer.word_index) + 1, VOCAB_SIZE + 1):,}"
    )

    return tokenizer


def _decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(
        img,
        channels=CHANNELS,
        expand_animations=False,
    )
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE],
    )
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img


def _augment_image(img, text):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.08)
    img = tf.image.random_contrast(
        img,
        lower=0.9,
        upper=1.1,
    )
    img = tf.clip_by_value(img, -1.0, 1.0)
    return img, text


def load_image_caption_dataset(rows, tokenizer):
    image_paths = [os.path.join(IMAGE_DIR, image_name) for image_name, _ in rows]
    captions = [caption for _, caption in rows]

    sequences = tokenizer.texts_to_sequences(captions)

    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="pre",
        truncating="pre",
    ).astype(np.int32)

    image_ds = tf.data.Dataset.from_tensor_slices(image_paths).map(
        _decode_image,
        num_parallel_calls=AUTOTUNE,
    )

    text_ds = tf.data.Dataset.from_tensor_slices(padded)

    dataset = tf.data.Dataset.zip((image_ds, text_ds))

    return (
        dataset.cache()
        .shuffle(
            min(1000, len(image_paths)),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        .map(
            _augment_image,
            num_parallel_calls=AUTOTUNE,
        )
        .batch(
            BATCH_SIZE,
            drop_remainder=True,
        )
        .prefetch(AUTOTUNE)
    )


def make_text_encoder():
    """Fresh text encoder trained with the V8 caption distribution."""
    token_input = tf.keras.Input(
        shape=(MAX_LEN,),
        dtype=tf.int32,
        name="caption_tokens",
    )

    x = layers.Embedding(
        input_dim=VOCAB_SIZE + 1,
        output_dim=EMBED_DIM,
        mask_zero=True,
        name="token_embedding",
    )(token_input)

    x = layers.LSTM(
        EMBED_DIM,
        name="caption_lstm",
    )(x)

    x = layers.LayerNormalization(
        name="caption_norm",
    )(x)

    return tf.keras.Model(
        token_input,
        x,
        name="text_encoder_v8",
    )


def make_generator():
    """Stable V4-style generator used as the V8 baseline."""
    noise_input = tf.keras.Input(
        shape=(NOISE_DIM,),
        name="noise_input",
    )
    text_input = tf.keras.Input(
        shape=(EMBED_DIM,),
        name="text_input",
    )

    ca = ConditioningAugmentation(
        EMBED_DIM,
        name="conditioning_augmentation",
    )(text_input)

    x = layers.Concatenate(name="noise_text_concat")([noise_input, ca])

    x = layers.Dense(
        5 * 5 * 128,
        use_bias=False,
        name="base_dense",
    )(x)
    x = layers.BatchNormalization(name="base_bn")(x)
    x = layers.LeakyReLU(0.2, name="base_activation")(x)
    x = layers.Reshape((5, 5, 128), name="base_reshape")(x)

    x = layers.UpSampling2D(name="upsample_10x10")(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv_64")(x)
    x = layers.BatchNormalization(name="bn_64")(x)
    x = layers.LeakyReLU(0.2, name="act_64")(x)

    x = layers.UpSampling2D(name="upsample_20x20")(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv_32")(x)
    x = layers.BatchNormalization(name="bn_32")(x)
    x = layers.LeakyReLU(0.2, name="act_32")(x)

    x = layers.UpSampling2D(name="upsample_40x40")(x)
    x = layers.Conv2D(16, 3, padding="same", use_bias=False, name="conv_16")(x)
    x = layers.BatchNormalization(name="bn_16")(x)
    x = layers.LeakyReLU(0.2, name="act_16")(x)

    x = layers.Resizing(
        IMG_SIZE,
        IMG_SIZE,
        name="resize_60x60",
    )(x)

    x = layers.Conv2D(8, 3, padding="same", use_bias=False, name="conv_8")(x)
    x = layers.BatchNormalization(name="bn_8")(x)
    x = layers.LeakyReLU(0.2, name="act_8")(x)

    output = layers.Conv2D(
        CHANNELS,
        3,
        padding="same",
        activation="tanh",
        name="output_conv",
    )(x)

    return tf.keras.Model(
        [noise_input, text_input],
        output,
        name="generator_v8",
    )


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
    compatibility = layers.Dot(
        axes=1,
        normalize=True,
        name="image_text_compatibility",
    )([image_features, text_features])
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


def make_mismatched_captions(caption_tokens):
    batch_size = tf.shape(caption_tokens)[0]

    return tf.cond(
        tf.greater(batch_size, 1),
        lambda: tf.roll(
            caption_tokens,
            shift=1,
            axis=0,
        ),
        lambda: caption_tokens,
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
        text_features = text_encoder(
            caption_tokens,
            training=True,
        )

        noise = tf.random.normal([batch_size, NOISE_DIM])

        fake_images = generator(
            [noise, text_features],
            training=True,
        )

        fake_images_for_d = tf.stop_gradient(fake_images)

        real_output = discriminator(
            [images, text_features],
            training=True,
        )

        fake_output = discriminator(
            [fake_images_for_d, text_features],
            training=True,
        )

        mismatched_tokens = make_mismatched_captions(caption_tokens)

        mismatched_features = text_encoder(
            mismatched_tokens,
            training=True,
        )

        mismatch_output = discriminator(
            [images, mismatched_features],
            training=True,
        )

        d_loss = discriminator_loss(
            real_output,
            fake_output,
        )

        mismatch_loss = tf.reduce_mean(tf.nn.relu(1.0 + mismatch_output))

        d_loss = d_loss + MISMATCH_LOSS_WEIGHT * mismatch_loss

    trainable_vars = (
        discriminator.trainable_variables + text_encoder.trainable_variables
    )

    grads = tape.gradient(
        d_loss,
        trainable_vars,
    )

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

        g_loss = generator_loss(fake_output)

    trainable_vars = generator.trainable_variables + text_encoder.trainable_variables

    grads = tape.gradient(
        g_loss,
        trainable_vars,
    )

    gen_opt.apply_gradients(zip(grads, trainable_vars))

    return g_loss


def update_ema(model, ema_model, decay):
    for v_ema, v in zip(
        ema_model.weights,
        model.weights,
    ):
        v_ema.assign(decay * v_ema + (1.0 - decay) * v)


def save_generated_samples(
    epoch_number,
    generator,
    monitor_noise,
    monitor_text,
):
    images = generator(
        [monitor_noise, monitor_text],
        training=False,
    )

    images = tf.clip_by_value(
        (images + 1.0) / 2.0,
        0.0,
        1.0,
    )

    n = int(images.shape[0])
    grid = int(np.ceil(np.sqrt(n)))

    if n < grid * grid:
        images = tf.concat(
            [
                images,
                tf.zeros(
                    [
                        grid * grid - n,
                        IMG_SIZE,
                        IMG_SIZE,
                        CHANNELS,
                    ],
                    dtype=images.dtype,
                ),
            ],
            axis=0,
        )

    tiled = tf.reshape(
        images,
        [
            grid,
            grid,
            IMG_SIZE,
            IMG_SIZE,
            CHANNELS,
        ],
    )

    tiled = tf.transpose(
        tiled,
        [0, 2, 1, 3, 4],
    )

    tiled = tf.reshape(
        tiled,
        [
            grid * IMG_SIZE,
            grid * IMG_SIZE,
            CHANNELS,
        ],
    )

    out_dir = os.path.join(
        V8_MODEL_DIR,
        "gen_images",
        "v8",
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    out_path = os.path.join(
        out_dir,
        f"generated_image_epoch_{epoch_number}.png",
    )

    tf.io.write_file(
        out_path,
        tf.io.encode_png(
            tf.cast(
                tiled * 255.0,
                tf.uint8,
            )
        ),
    )

    print(f"Saved samples: {out_path}")


def main():
    configure_runtime()

    os.makedirs(
        V8_MODEL_DIR,
        exist_ok=True,
    )
    os.makedirs(
        V8_CHECKPOINT_DIR,
        exist_ok=True,
    )

    rows = prepare_v8_captions()
    tokenizer = build_tokenizer(rows)

    with open(
        TOKENIZER_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(tokenizer.to_json())

    payload = load_image_caption_dataset(
        rows,
        tokenizer,
    )

    text_encoder = make_text_encoder()
    generator = make_generator()
    discriminator = make_discriminator()

    # Build all models before creating the EMA model/checkpoint.
    dummy_tokens = tf.zeros(
        [1, MAX_LEN],
        dtype=tf.int32,
    )
    dummy_text = text_encoder(
        dummy_tokens,
        training=False,
    )
    dummy_noise = tf.zeros(
        [1, NOISE_DIM],
        dtype=tf.float32,
    )

    generator(
        [dummy_noise, dummy_text],
        training=False,
    )

    discriminator(
        [
            tf.zeros([1, IMG_SIZE, IMG_SIZE, CHANNELS]),
            dummy_text,
        ],
        training=False,
    )

    g_ema = make_generator()
    g_ema(
        [dummy_noise, dummy_text],
        training=False,
    )
    g_ema.set_weights(generator.get_weights())
    g_ema.trainable = False

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

    epoch_var = tf.Variable(
        0,
        dtype=tf.int64,
        trainable=False,
        name="v8_epoch",
    )

    checkpoint_prefix = os.path.join(
        V8_CHECKPOINT_DIR,
        "ckpt_60px_conditional_v8",
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

    latest_ckpt = tf.train.latest_checkpoint(V8_CHECKPOINT_DIR)

    if latest_ckpt:
        checkpoint.restore(latest_ckpt).expect_partial()
        print(f"Resumed V8 checkpoint: {latest_ckpt}")
    else:
        print("Starting V8 from scratch with the current captions.txt.")

    print()
    print("V8 configuration")
    print("================")
    print(f"Pairs used: {len(rows)}")
    print(f"Generator parameters: " f"{generator.count_params():,}")
    print(f"Discriminator parameters: " f"{discriminator.count_params():,}")
    print(f"Text encoder parameters: " f"{text_encoder.count_params():,}")
    print("Generator architecture: stable V4-style baseline")
    print(f"Mismatch loss weight: " f"{MISMATCH_LOSS_WEIGHT:.2f}")
    print(f"Training schedule: epochs " f"{int(epoch_var.numpy()) + 1}-{EPOCHS}")

    monitor_noise = tf.random.stateless_normal(
        [MONITOR_GRID, NOISE_DIM],
        seed=[SEED, 0],
    )

    monitor_tokens = tf.constant(
        pad_sequences(
            tokenizer.texts_to_sequences(MONITOR_CAPTIONS),
            maxlen=MAX_LEN,
            padding="pre",
            truncating="pre",
        ).astype(np.int32),
        dtype=tf.int32,
    )

    monitor_text = text_encoder(
        monitor_tokens,
        training=False,
    )

    g_metric = tf.keras.metrics.Mean()
    d_metric = tf.keras.metrics.Mean()

    while int(epoch_var.numpy()) < EPOCHS:
        g_metric.reset_state()
        d_metric.reset_state()

        epoch_start = time.time()
        actual_epoch = int(epoch_var.numpy()) + 1

        for step_idx, (
            image_batch,
            caption_batch,
        ) in enumerate(payload):

            d_loss = disc_step(
                image_batch,
                caption_batch,
                text_encoder,
                generator,
                discriminator,
                disc_opt,
            )

            d_metric.update_state(d_loss)

            if step_idx % N_CRITIC == 0:
                g_loss = gen_step(
                    caption_batch,
                    text_encoder,
                    generator,
                    discriminator,
                    gen_opt,
                )

                g_metric.update_state(g_loss)

                if step_idx % EMA_UPDATE_EVERY == 0:
                    update_ema(
                        generator,
                        g_ema,
                        EMA_DECAY,
                    )

            if (step_idx + 1) % LOG_EVERY_STEPS == 0:
                sys.stdout.write(
                    f"\rEpoch {actual_epoch:>4}  "
                    f"step {step_idx + 1:>5}  "
                    f"g={float(g_metric.result()):.4f}  "
                    f"d={float(d_metric.result()):.4f}"
                )
                sys.stdout.flush()

        epoch_var.assign(actual_epoch)

        elapsed = time.time() - epoch_start
        g_avg = float(g_metric.result())
        d_avg = float(d_metric.result())

        sys.stdout.write("\r" + " " * 100 + "\r")

        print(
            f"Epoch {actual_epoch}/{EPOCHS}  "
            f"Gen {g_avg:.4f}  "
            f"Disc {d_avg:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if actual_epoch % 10 == 0 or actual_epoch == EPOCHS:
            save_generated_samples(
                actual_epoch,
                g_ema,
                monitor_noise,
                monitor_text,
            )

            checkpoint.save(file_prefix=checkpoint_prefix)

    generator.save(GENERATOR_PATH)
    g_ema.save(EMA_GENERATOR_PATH)
    discriminator.save(DISCRIMINATOR_PATH)
    text_encoder.save(TEXT_ENCODER_PATH)

    # Keep the exact cleaned captions used for training and the tokenizer
    # together with the V8 model.
    with open(
        TOKENIZER_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(tokenizer.to_json())

    print()
    print("V8 training complete.")
    print(f"Models saved in: {V8_MODEL_DIR}")
    print(f"Training captions saved in: " f"{CLEAN_CAPTIONS_PATH}")


if __name__ == "__main__":
    main()
