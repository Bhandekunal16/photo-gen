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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


IMG_SIZE = 60
CHANNELS = 3
BATCH_SIZE = 32
NOISE_DIM = 256
EPOCHS = 300
MAX_LEN = 20
VOCAB_SIZE = 5000
EMBED_DIM = 256
DISC_TEXT_DIM = 128
N_CRITIC = 1
EMA_DECAY = 0.995
EMA_UPDATE_EVERY = 5
MONITOR_GRID = 4
LOG_EVERY_STEPS = 20


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
embedding_layer = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM)


class ConditioningAugmentation(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_mean = layers.Dense(embed_dim)
        self.dense_log_sigma = layers.Dense(embed_dim)

    def call(self, inputs):
        mean = self.dense_mean(inputs)
        log_sigma = self.dense_log_sigma(inputs)
        stddev = tf.exp(log_sigma)
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

    text_lstm = tf.keras.layers.LSTM(EMBED_DIM)
    embedded = embedding_layer(tf.constant(padded))
    text_features = text_lstm(embedded).numpy().astype(np.float32)

    image_ds = tf.data.Dataset.from_tensor_slices(image_paths).map(
        _decode_image, num_parallel_calls=AUTOTUNE
    )
    text_ds = tf.data.Dataset.from_tensor_slices(text_features)

    dataset = tf.data.Dataset.zip((image_ds, text_ds))
    return (
        dataset.cache()
        .shuffle(min(1000, len(image_paths)), reshuffle_each_iteration=True)
        .map(_augment_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )


def make_generator():
    noise_input = tf.keras.Input(shape=(NOISE_DIM,))
    text_input = tf.keras.Input(shape=(EMBED_DIM,))

    ca = ConditioningAugmentation(EMBED_DIM)(text_input)
    x = layers.Concatenate()([noise_input, ca])

    x = layers.Dense(5 * 5 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((5, 5, 256))(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Resizing(60, 60)(x)
    x = layers.Conv2D(16, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(
        CHANNELS, kernel_size=3, padding="same", use_bias=False, activation="tanh"
    )(x)

    return tf.keras.Model([noise_input, text_input], x)


def make_discriminator():
    image_input = tf.keras.Input(shape=(60, 60, 3))
    text_input = tf.keras.Input(shape=(EMBED_DIM,))

    x = layers.Conv2D(32, 4, strides=2, padding="same")(image_input)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(64, 4, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, 4, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)

    text_proj = layers.Dense(DISC_TEXT_DIM, activation="relu")(text_input)
    ca_text = ConditioningAugmentation(DISC_TEXT_DIM)(text_proj)

    x = layers.Concatenate()([x, ca_text])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(1)(x)

    return tf.keras.Model([image_input, text_input], x)


def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(tf.nn.relu(1.0 - real_output)) + tf.reduce_mean(
        tf.nn.relu(1.0 + fake_output)
    )


def gradient_penalty(discriminator, real_images, fake_images, text_embeddings):
    batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated_images = alpha * real_images + (1 - alpha) * fake_images
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        interpolated_output = discriminator(
            [interpolated_images, text_embeddings], training=True
        )
    grads = tape.gradient(interpolated_output, [interpolated_images])[0]
    grads_sq = tf.reduce_sum(tf.square(grads), axis=[1, 2, 3])
    grad_norm = tf.sqrt(grads_sq + 1e-12)
    penalty = tf.reduce_mean((grad_norm - 1.0) ** 2)
    return penalty * 10


@tf.function(reduce_retracing=True)
def disc_step(images, captions, generator, discriminator, disc_opt):
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, NOISE_DIM])
    fake_images = generator([noise, captions], training=True)

    with tf.GradientTape() as tape:
        real_output = discriminator([images, captions], training=True)
        fake_output = discriminator([fake_images, captions], training=True)
        d_loss = discriminator_loss(real_output, fake_output)

    grads = tape.gradient(d_loss, discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(grads, discriminator.trainable_variables))
    return d_loss


@tf.function(reduce_retracing=True)
def gen_step(captions, generator, discriminator, gen_opt):
    """One generator update."""
    batch_size = tf.shape(captions)[0]
    noise = tf.random.normal([batch_size, NOISE_DIM])

    with tf.GradientTape() as tape:
        fake_images = generator([noise, captions], training=True)
        fake_output = discriminator([fake_images, captions], training=True)
        g_loss = generator_loss(fake_output)

    grads = tape.gradient(g_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(grads, generator.trainable_variables))
    return g_loss


def build_ema_generator(generator):
    ema = make_generator()
    ema.set_weights(generator.get_weights())
    ema.trainable = False
    return ema


@tf.function
def update_ema(model, ema_model, decay):
    for v_ema, v in zip(ema_model.weights, model.weights):
        v_ema.assign(decay * v_ema + (1.0 - decay) * v)


def save_generated_samples(epoch, folder_type, generator, monitor_noise, monitor_text):
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

    out_dir = f"gen_images/{folder_type}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"generated_image_epoch_{epoch + 1}.png")
    tf.io.write_file(out_path, tf.io.encode_png(tiled_u8))


def _resolve_captions_path() -> str:
    v2 = "./data/captions_60px_v2.txt"
    v1 = "./data/captions_60px.txt"
    return v2 if os.path.exists(v2) else v1


def main():
    configure_runtime()

    captions_path = _resolve_captions_path()
    print(f"Using captions file: {captions_path}")
    with open(captions_path) as f:
        all_captions = [
            line.strip().split("|", 1)[1] for line in f if line.strip() and "|" in line
        ]

    tokenizer.fit_on_texts(all_captions)
    payload = load_image_caption_dataset("./data/image60px", captions_path)
    discriminator = make_discriminator()
    generator = make_generator()
    g_ema = build_ema_generator(generator)
    gen_opt = tf.keras.optimizers.Adam(3e-4, beta_1=0.5, beta_2=0.999)
    disc_opt = tf.keras.optimizers.Adam(2.5e-5, beta_1=0.5, beta_2=0.9)

    checkpoint_dir = "./checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_60px")
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        g_ema=g_ema,
        g_optimizer=gen_opt,
        d_optimizer=disc_opt,
    )

    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        checkpoint.restore(latest_ckpt).expect_partial()
        print(f"Restored from checkpoint: {latest_ckpt}")

        gen_opt.learning_rate.assign(3e-4)
        disc_opt.learning_rate.assign(2.5e-5)
        print(
            f"Forced LR override: gen={float(gen_opt.learning_rate):.1e}, "
            f"disc={float(disc_opt.learning_rate):.1e}"
        )
    else:
        print("Starting training from scratch.")

    monitor_noise = tf.random.normal([MONITOR_GRID, NOISE_DIM])
    seq = tokenizer.texts_to_sequences(["a body of water"] * MONITOR_GRID)
    padded = pad_sequences(seq, maxlen=MAX_LEN).astype(np.int32)

    monitor_text = tf.reduce_mean(embedding_layer(tf.constant(padded)), axis=1)

    g_loss_metric = tf.keras.metrics.Mean()
    d_loss_metric = tf.keras.metrics.Mean()

    for epoch in range(EPOCHS):
        g_loss_metric.reset_state()
        d_loss_metric.reset_state()
        epoch_start = time.time()

        for step_idx, (image_batch, caption_batch) in enumerate(payload):
            d_loss = disc_step(
                image_batch, caption_batch, generator, discriminator, disc_opt
            )
            d_loss_metric.update_state(d_loss)

            if step_idx % N_CRITIC == 0:
                g_loss = gen_step(caption_batch, generator, discriminator, gen_opt)
                g_loss_metric.update_state(g_loss)
                if step_idx % EMA_UPDATE_EVERY == 0:
                    update_ema(generator, g_ema, EMA_DECAY)

            if (step_idx + 1) % LOG_EVERY_STEPS == 0:
                sys.stdout.write(
                    f"\rEpoch {epoch+1:>4}  step {step_idx+1:>5}  "
                    f"g={float(g_loss_metric.result()):.4f}  "
                    f"d={float(d_loss_metric.result()):.4f}"
                )
                sys.stdout.flush()

        elapsed = time.time() - epoch_start
        g_avg = float(g_loss_metric.result())
        d_avg = float(d_loss_metric.result())

        sys.stdout.write("\r" + " " * 80 + "\r")
        print(
            f"Epoch {epoch+1}/{EPOCHS}  Gen {g_avg:.4f}  Disc {d_avg:.4f}  ({elapsed:.1f}s)"
        )

        if g_avg <= 0.8 and d_avg <= 0.8:
            save_generated_samples(epoch, "perfect", g_ema, monitor_noise, monitor_text)

        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            save_generated_samples(epoch, "normal", g_ema, monitor_noise, monitor_text)

        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    generator.save("generator_model_60px.keras")
    g_ema.save("generator_ema_model_60px.keras")
    discriminator.save("discriminator_model_60px.keras")


if __name__ == "__main__":
    main()
