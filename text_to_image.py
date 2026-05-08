import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration & Constants ---
IMG_SIZE = 60
CHANNELS = 3
BATCH_SIZE = 32
NOISE_DIM = 256
EPOCHS = 2000
MAX_LEN = 20
VOCAB_SIZE = 5000
EMBED_DIM = 256
N_CRITIC = 2

# Performance toggles. Mixed precision can ~2x throughput on modern GPUs
# but WGAN-GP is sensitive, so it's opt-in.
USE_MIXED_PRECISION = False
USE_XLA = True
AUTOTUNE = tf.data.AUTOTUNE


def configure_runtime():
    """Enable GPU memory growth, XLA, and (optionally) mixed precision once."""
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if USE_XLA:
        tf.config.optimizer.set_jit(True)

    if USE_MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


# --- Global Layers & Tokenizer Setup ---
# These are initialized globally as they are used across multiple functions
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
embedding_layer = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM)

# --- Classes ---
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

# --- Helper Functions ---
def _decode_image(path):
    """Read + decode + resize + normalize a single image. Runs in graph mode."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=CHANNELS)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img


def load_image_caption_dataset(img_folder, caption_file):
    """Build a fast tf.data pipeline.

    Optimizations vs. the original:
      * Captions are tokenized once in NumPy (no per-sample tf.py_function).
      * Text features are computed once in a single batched LSTM call instead
        of inside dataset.map (the embedding/LSTM weights aren't trained anyway).
      * Image decode runs in parallel via num_parallel_calls=AUTOTUNE.
      * Decoded tensors are cached in RAM so we only pay the JPEG cost once.
    """
    image_paths = []
    captions = []

    with open(caption_file, "r") as f:
        for line in f:
            img_name, caption = line.strip().split("|")
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
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )

def make_generator():
    noise_input = tf.keras.Input(shape=(NOISE_DIM,))
    text_input = tf.keras.Input(shape=(EMBED_DIM,))

    ca = ConditioningAugmentation(EMBED_DIM)(text_input)
    x = layers.Concatenate()([noise_input, ca])

    x = layers.Dense(5 * 5 * 512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((5, 5, 512))(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(256, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Resizing(60, 60)(x)
    x = layers.Conv2D(32, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(
        CHANNELS, kernel_size=3, padding="same", use_bias=False, activation="tanh"
    )(x)

    return tf.keras.Model([noise_input, text_input], x)

def make_discriminator():
    image_input = tf.keras.Input(shape=(60, 60, 3))
    text_input = tf.keras.Input(shape=(EMBED_DIM,))

    x = layers.Conv2D(64, 4, strides=2, padding="same")(image_input)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(512, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)

    text_proj = layers.Dense(x.shape[-1], activation="relu")(text_input)
    ca_text = ConditioningAugmentation(x.shape[-1])(text_proj)

    x = layers.Concatenate()([x, ca_text])
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(1)(x)

    return tf.keras.Model([image_input, text_input], x)

# --- Loss and Training Steps ---
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
    # Per-sample L2 norm (correct WGAN-GP form). The original used a single
    # scalar norm over the whole batch which is both wrong and slower to
    # converge.
    grads_sq = tf.reduce_sum(tf.square(grads), axis=[1, 2, 3])
    grad_norm = tf.sqrt(grads_sq + 1e-12)
    penalty = tf.reduce_mean((grad_norm - 1.0) ** 2)
    return penalty * 10


@tf.function(reduce_retracing=True)
def train_step(images, captions, generator, discriminator, gen_opt, disc_opt):
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator([noise, captions], training=True)

        real_output = discriminator([images, captions], training=True)
        fake_output = discriminator([fake_images, captions], training=True)

        gp = gradient_penalty(discriminator, images, fake_images, captions)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output) + gp

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    gen_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def save_generated_image(epoch, folder_type, generator):
    test_caption = "a body of water"
    seq = tokenizer.texts_to_sequences([test_caption])[0]
    padded = pad_sequences([seq], maxlen=MAX_LEN)
    padded_tensor = tf.constant(padded, dtype=tf.int32)

    noise = tf.random.normal([1, NOISE_DIM])
    embedding = embedding_layer(padded_tensor)
    embedding_mean = tf.reduce_mean(embedding, axis=1)

    generated = generator([noise, embedding_mean], training=False)
    img = (generated[0] + 1.0) / 2.0
    img_uint8 = tf.cast(tf.clip_by_value(img * 255.0, 0.0, 255.0), tf.uint8)

    out_dir = f"gen_images/{folder_type}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"generated_image_epoch_{epoch + 1}.png")
    # tf.io.encode_png is dramatically faster than matplotlib + savefig.
    tf.io.write_file(out_path, tf.io.encode_png(img_uint8))

# --- Main Training Loop ---
def main():
    configure_runtime()

    # 1. Prepare Data
    with open("./data/captions_60px.txt") as f:
        all_captions = [line.strip().split("|")[1] for line in f]
    
    tokenizer.fit_on_texts(all_captions)
    payload = load_image_caption_dataset("./data/image60px", "./data/captions_60px.txt")

    # 2. Build Models
    discriminator = make_discriminator()
    generator = make_generator()

    # 3. Optimizers
    gen_opt = tf.keras.optimizers.Adam(3e-4, beta_1=0.5, beta_2=0.999)
    disc_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    # 4. Checkpoints
    checkpoint_dir = "./checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_60px")
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=gen_opt,
        d_optimizer=disc_opt,
    )

    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        checkpoint.restore(latest_ckpt)
        print(f"Restored from checkpoint: {latest_ckpt}")
    else:
        print("Starting training from scratch.")

    # 5. Training Loop
    for epoch in range(EPOCHS):
        for image_batch, caption_batch in payload:
            g_loss, d_loss = train_step(
                image_batch, caption_batch, generator, discriminator, gen_opt, disc_opt
            )

        # .numpy() once per epoch to avoid syncing the device on every step.
        g_loss_v = float(g_loss.numpy())
        d_loss_v = float(d_loss.numpy())
        print(f"Epoch {epoch+1}, Gen Loss: {g_loss_v:.4f}, Disc Loss: {d_loss_v:.4f}")

        if g_loss_v <= 0.8 and d_loss_v <= 0.8:
            save_generated_image(epoch, 'perfect', generator)

        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            save_generated_image(epoch, 'normal', generator)

        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    # 6. Final Save
    generator.save("generator_model_60px.keras")
    discriminator.save("discriminator_model_60px.keras")

if __name__ == "__main__":
    main()