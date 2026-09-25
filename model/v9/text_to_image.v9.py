#!/usr/bin/env python3
"""
V9 - Text-conditioned GAN for 60x60 images.

Goal:
- Keep the stable V8/V4-style generator baseline.
- Strengthen caption -> image conditioning without replacing the whole GAN.
- Inject the same trainable text embedding at multiple generator resolutions.
- Use identity-initialized FiLM blocks so conditioning starts conservatively.
- Keep projection discriminator + mismatched-caption supervision.
- Train from scratch with the current data/captions.txt.
- Do not modify data/captions.txt; save the exact cleaned captions used by V9.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

import re
import json
import time
import random
import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------------------------------------------------------------
# Reproducibility / CPU
# ---------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CPU_INTRA_OP_THREADS = 4
CPU_INTER_OP_THREADS = 2
try:
    tf.config.threading.set_intra_op_parallelism_threads(CPU_INTRA_OP_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(CPU_INTER_OP_THREADS)
except RuntimeError:
    pass


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
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
SAVE_EVERY_EPOCHS = 10

DATA_DIR = Path("./data")
CAPTIONS_FILE = DATA_DIR / "captions.txt"
IMAGE_DIR = DATA_DIR / "image60px"

MODEL_DIR = Path("./model/v9")
DATA_OUT_DIR = MODEL_DIR / "data"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
SAMPLES_DIR = MODEL_DIR / "gen_images" / "v9"

TOKENIZER_PATH = MODEL_DIR / "tokenizer_60px_v9.json"
CAPTION_FINGERPRINT_PATH = MODEL_DIR / "caption_fingerprint_v9.txt"
GENERATOR_PATH = MODEL_DIR / "generator_model_60px_v9.keras"
EMA_GENERATOR_PATH = MODEL_DIR / "generator_ema_model_60px_v9.keras"
DISCRIMINATOR_PATH = MODEL_DIR / "discriminator_model_60px_v9.keras"
TEXT_ENCODER_PATH = MODEL_DIR / "text_encoder_60px_v9.keras"

for d in (MODEL_DIR, DATA_OUT_DIR, CHECKPOINT_DIR, SAMPLES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Caption preparation
# ---------------------------------------------------------------------
def normalize_caption(caption: str) -> str:
    caption = str(caption).strip().lower()
    caption = re.sub(r"\s+", " ", caption)
    caption = re.sub(r"([!?.,;:])\1+", r"\1", caption)

    words = caption.split()
    deduped = []
    for word in words:
        if not deduped or word != deduped[-1]:
            deduped.append(word)

    caption = " ".join(deduped)
    caption = re.sub(r"[^\w\s'-]", " ", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


def load_caption_pairs():
    if not CAPTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing captions file: {CAPTIONS_FILE}")
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Missing image directory: {IMAGE_DIR}")

    pairs = []
    rejected = []
    normalized_count = 0

    with CAPTIONS_FILE.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue

            if "|" not in raw:
                rejected.append((line_no, "missing | separator", raw))
                continue

            image_name, caption = raw.split("|", 1)
            image_name = image_name.strip()
            original_caption = caption.strip()
            caption = normalize_caption(original_caption)

            if caption != original_caption.lower().strip():
                normalized_count += 1

            if not image_name or not caption:
                rejected.append((line_no, "empty image/caption", raw))
                continue

            if caption in {"thumb", "thumbnail"}:
                rejected.append((line_no, "thumbnail caption", raw))
                continue

            if len(caption.split()) < 2:
                rejected.append((line_no, "caption too short", raw))
                continue

            image_path = IMAGE_DIR / image_name
            if not image_path.exists():
                rejected.append((line_no, "image file missing", raw))
                continue

            pairs.append((str(image_path), caption))

    if not pairs:
        raise RuntimeError("No usable image-caption pairs found.")

    used_path = DATA_OUT_DIR / "captions_used_v9.txt"
    with used_path.open("w", encoding="utf-8") as f:
        for image_path, caption in pairs:
            f.write(f"{Path(image_path).name}|{caption}\n")

    fingerprint_text = "\n".join(
        f"{Path(image_path).name}|{caption}"
        for image_path, caption in pairs
    )
    fingerprint = hashlib.sha256(
        fingerprint_text.encode("utf-8")
    ).hexdigest()
    CAPTION_FINGERPRINT_PATH.write_text(
        fingerprint,
        encoding="utf-8",
    )

    print()
    print("V9 caption preparation")
    print("=======================")
    print(f"Original pairs:       {len(pairs) + len(rejected)}")
    print(f"Usable pairs:         {len(pairs)}")
    print(f"Rejected pairs:       {len(rejected)}")
    print(f"Normalized captions:  {normalized_count}")
    print(f"Used caption file:    {used_path}")
    if rejected:
        print("Rejected examples:")
        for item in rejected[:5]:
            print(f"  line {item[0]}: {item[1]}")

    return pairs


# ---------------------------------------------------------------------
# Tokenizer / text encoder
# ---------------------------------------------------------------------
def build_tokenizer(captions):
    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token="<unk>",
        lower=True,
    )
    tokenizer.fit_on_texts(captions)

    with TOKENIZER_PATH.open("w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    print(f"Tokenizer vocabulary: {min(len(tokenizer.word_index) + 1, VOCAB_SIZE)}")
    return tokenizer


def make_text_encoder():
    token_input = tf.keras.Input(
        shape=(MAX_LEN,),
        dtype=tf.int32,
        name="caption_tokens",
    )

    x = layers.Embedding(
        input_dim=VOCAB_SIZE + 1,
        output_dim=EMBED_DIM,
        mask_zero=True,
        name="caption_embedding",
    )(token_input)

    x = layers.LSTM(
        EMBED_DIM,
        name="caption_lstm",
    )(x)

    x = layers.LayerNormalization(
        name="caption_normalization",
    )(x)

    return Model(token_input, x, name="text_encoder_v9")


# ---------------------------------------------------------------------
# Conditioning augmentation
# ---------------------------------------------------------------------
class ConditioningAugmentation(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.mu = layers.Dense(dim, name="ca_mu")
        self.log_sigma = layers.Dense(dim, name="ca_log_sigma")

    def call(self, text_embedding, training=None):
        mu = self.mu(text_embedding)
        log_sigma = tf.clip_by_value(self.log_sigma(text_embedding), -4.0, 4.0)

        if training:
            epsilon = tf.random.normal(tf.shape(mu))
            return mu + tf.exp(0.5 * log_sigma) * epsilon

        return mu


class IdentityFiLM(layers.Layer):
    """
    Multi-stage text conditioning.

    The Dense kernel and bias are zero initialized, making the layer
    initially behave approximately as identity:
        y = x * (1 + 0) + 0

    The conditioning can then grow during training rather than
    disrupting the pretrained/stable visual path at initialization.
    """

    def __init__(self, channels, strength=0.5, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.strength = strength

        self.to_gamma_beta = layers.Dense(
            channels * 2,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="film_parameters",
        )

    def call(self, inputs):
        x, text_embedding = inputs
        params = self.to_gamma_beta(text_embedding)

        gamma, beta = tf.split(params, 2, axis=-1)

        gamma = self.strength * tf.tanh(gamma)
        beta = self.strength * tf.tanh(beta)

        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]

        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------
def make_generator():
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

    x = layers.Concatenate(name="noise_text_concat")(
        [noise_input, ca]
    )

    x = layers.Dense(
        5 * 5 * 128,
        use_bias=False,
        name="dense_5x5",
    )(x)
    x = layers.BatchNormalization(name="bn_5x5")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((5, 5, 128), name="reshape_5x5")(x)

    x = IdentityFiLM(128, name="film_5x5")([x, text_input])

    x = layers.UpSampling2D(name="up_10x10")(x)
    x = layers.Conv2D(
        64, 3, padding="same", use_bias=False, name="conv_10x10"
    )(x)
    x = layers.BatchNormalization(name="bn_10x10")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = IdentityFiLM(64, name="film_10x10")([x, text_input])

    x = layers.UpSampling2D(name="up_20x20")(x)
    x = layers.Conv2D(
        32, 3, padding="same", use_bias=False, name="conv_20x20"
    )(x)
    x = layers.BatchNormalization(name="bn_20x20")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = IdentityFiLM(32, name="film_20x20")([x, text_input])

    x = layers.UpSampling2D(name="up_40x40")(x)
    x = layers.Conv2D(
        16, 3, padding="same", use_bias=False, name="conv_40x40"
    )(x)
    x = layers.BatchNormalization(name="bn_40x40")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = IdentityFiLM(16, name="film_40x40")([x, text_input])

    x = layers.Resizing(IMG_SIZE, IMG_SIZE, name="resize_60x60")(x)
    x = layers.Conv2D(
        8, 3, padding="same", use_bias=False, name="conv_60x60"
    )(x)
    x = layers.BatchNormalization(name="bn_60x60")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = IdentityFiLM(8, name="film_60x60")([x, text_input])

    output = layers.Conv2D(
        CHANNELS,
        3,
        padding="same",
        activation="tanh",
        name="rgb_output",
    )(x)

    return Model(
        [noise_input, text_input],
        output,
        name="generator_v9",
    )


# ---------------------------------------------------------------------
# Projection discriminator
# ---------------------------------------------------------------------
def make_discriminator():
    image_input = tf.keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
        name="image_input",
    )
    text_input = tf.keras.Input(
        shape=(EMBED_DIM,),
        name="text_input",
    )

    x = layers.Conv2D(
        32, 4, strides=2, padding="same", name="disc_conv_1"
    )(image_input)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(
        64, 4, strides=2, padding="same", name="disc_conv_2"
    )(x)
    x = layers.LayerNormalization(name="disc_ln_2")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(
        128, 4, strides=2, padding="same", name="disc_conv_3"
    )(x)
    x = layers.LayerNormalization(name="disc_ln_3")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(
        128, 4, strides=2, padding="same", name="disc_conv_4"
    )(x)
    x = layers.LayerNormalization(name="disc_ln_4")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.GlobalAveragePooling2D(name="disc_gap")(x)

    image_features = layers.Dense(
        EMBED_DIM,
        name="image_projection",
    )(x)

    text_features = layers.Dense(
        EMBED_DIM,
        name="text_projection",
    )(text_input)

    compatibility = layers.Dot(
        axes=1,
        normalize=True,
        name="image_text_compatibility",
    )([image_features, text_features])

    realness = layers.Dense(
        1,
        name="realness",
    )(x)

    output = layers.Add(
        name="conditional_score",
    )([realness, compatibility])

    return Model(
        [image_input, text_input],
        output,
        name="conditional_discriminator_v9",
    )


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
def decode_image(path):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(
        image_bytes,
        channels=CHANNELS,
        expand_animations=False,
    )
    image.set_shape([None, None, CHANNELS])
    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
    )
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    return image


def make_dataset(pairs, tokenizer):
    image_paths = [p[0] for p in pairs]
    captions = [p[1] for p in pairs]

    token_ids = tokenizer.texts_to_sequences(captions)
    token_ids = pad_sequences(
        token_ids,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    ).astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices(
        (image_paths, token_ids)
    )

    ds = ds.shuffle(
        max(len(pairs), 128),
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    def load(path, tokens):
        return decode_image(path), tokens

    ds = ds.map(
        load,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(
        BATCH_SIZE,
        drop_remainder=True,
    )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def disc_step(
    real_images,
    caption_tokens,
    noise,
    generator,
    discriminator,
    text_encoder,
    optimizer,
):
    with tf.GradientTape() as tape:
        text_embedding = text_encoder(
            caption_tokens,
            training=True,
        )

        fake_images = generator(
            [noise, text_embedding],
            training=True,
        )

        real_output = discriminator(
            [real_images, text_embedding],
            training=True,
        )

        fake_output = discriminator(
            [tf.stop_gradient(fake_images), text_embedding],
            training=True,
        )

        if tf.shape(caption_tokens)[0] > 1:
            mismatched_tokens = tf.roll(
                caption_tokens,
                shift=1,
                axis=0,
            )
        else:
            mismatched_tokens = caption_tokens

        mismatch_embedding = text_encoder(
            mismatched_tokens,
            training=True,
        )

        mismatch_output = discriminator(
            [real_images, mismatch_embedding],
            training=True,
        )

        real_loss = tf.reduce_mean(
            tf.nn.relu(1.0 - real_output)
        )

        fake_loss = tf.reduce_mean(
            tf.nn.relu(1.0 + fake_output)
        )

        mismatch_loss = tf.reduce_mean(
            tf.nn.relu(1.0 + mismatch_output)
        )

        total_loss = (
            real_loss
            + fake_loss
            + MISMATCH_LOSS_WEIGHT * mismatch_loss
        )

    variables = (
        discriminator.trainable_variables
        + text_encoder.trainable_variables
    )

    gradients = tape.gradient(total_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return total_loss


def gen_step(
    caption_tokens,
    noise,
    generator,
    discriminator,
    text_encoder,
    optimizer,
):
    with tf.GradientTape() as tape:
        text_embedding = text_encoder(
            caption_tokens,
            training=True,
        )

        fake_images = generator(
            [noise, text_embedding],
            training=True,
        )

        fake_output = discriminator(
            [fake_images, text_embedding],
            training=False,
        )

        loss = -tf.reduce_mean(fake_output)

    variables = (
        generator.trainable_variables
        + text_encoder.trainable_variables
    )

    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


# ---------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------
def build_ema_generator(generator):
    ema_generator = make_generator()
    ema_generator.set_weights(generator.get_weights())
    return ema_generator


def update_ema(ema_generator, generator, decay):
    ema_weights = ema_generator.get_weights()
    gen_weights = generator.get_weights()

    updated = [
        decay * old + (1.0 - decay) * new
        for old, new in zip(ema_weights, gen_weights)
    ]

    ema_generator.set_weights(updated)


# ---------------------------------------------------------------------
# Fixed monitoring captions
# ---------------------------------------------------------------------
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


def make_monitor_tokens(tokenizer):
    token_ids = tokenizer.texts_to_sequences(MONITOR_CAPTIONS)
    token_ids = pad_sequences(
        token_ids,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    ).astype(np.int32)
    return np.asarray(token_ids)


def save_sample_grid(generator, text_encoder, tokenizer, epoch):
    tokens = make_monitor_tokens(tokenizer)

    rng = np.random.default_rng(SEED)
    noise = rng.normal(
        0.0,
        1.0,
        size=(MONITOR_GRID, NOISE_DIM),
    ).astype(np.float32)

    embeddings = text_encoder(
        tf.convert_to_tensor(tokens),
        training=False,
    )

    images = generator(
        [tf.convert_to_tensor(noise), embeddings],
        training=False,
    ).numpy()

    images = np.clip((images + 1.0) * 127.5, 0, 255).astype(np.uint8)

    rows = int(np.sqrt(MONITOR_GRID))
    cols = int(np.ceil(MONITOR_GRID / rows))

    canvas = np.zeros(
        (rows * IMG_SIZE, cols * IMG_SIZE, CHANNELS),
        dtype=np.uint8,
    )

    for i in range(MONITOR_GRID):
        r = i // cols
        c = i % cols
        canvas[
            r * IMG_SIZE:(r + 1) * IMG_SIZE,
            c * IMG_SIZE:(c + 1) * IMG_SIZE,
        ] = images[i]

    path = SAMPLES_DIR / f"generated_image_epoch_{epoch}.png"

    tf.keras.utils.save_img(
        str(path),
        canvas,
    )

    print(f"Saved samples: {path}")


def save_caption_sweep(generator, text_encoder, tokenizer, epoch):
    # Diagnostic: use one identical noise vector for every caption.
    # If text conditioning works, changing only the caption should
    # visibly change the generated image.
    tokens = make_monitor_tokens(tokenizer)

    noise = np.zeros((MONITOR_GRID, NOISE_DIM), dtype=np.float32)
    noise[0] = np.random.default_rng(SEED).normal(
        0.0, 1.0, size=(NOISE_DIM,)
    ).astype(np.float32)
    for i in range(1, MONITOR_GRID):
        noise[i] = noise[0]

    embeddings = text_encoder(
        tf.convert_to_tensor(tokens),
        training=False,
    )

    images = generator(
        [tf.convert_to_tensor(noise), embeddings],
        training=False,
    ).numpy()

    images = np.clip((images + 1.0) * 127.5, 0, 255).astype(np.uint8)

    rows = int(np.sqrt(MONITOR_GRID))
    cols = int(np.ceil(MONITOR_GRID / rows))
    canvas = np.zeros(
        (rows * IMG_SIZE, cols * IMG_SIZE, CHANNELS),
        dtype=np.uint8,
    )

    for i in range(MONITOR_GRID):
        r = i // cols
        c = i % cols
        canvas[
            r * IMG_SIZE:(r + 1) * IMG_SIZE,
            c * IMG_SIZE:(c + 1) * IMG_SIZE,
        ] = images[i]

    path = SAMPLES_DIR / f"caption_sweep_epoch_{epoch}.png"
    tf.keras.utils.save_img(str(path), canvas)

    caption_path = SAMPLES_DIR / f"caption_sweep_epoch_{epoch}.txt"
    caption_path.write_text(
        "\n".join(
            f"{i + 1}. {caption}"
            for i, caption in enumerate(MONITOR_CAPTIONS)
        ),
        encoding="utf-8",
    )

    print(f"Saved caption sweep: {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    pairs = load_caption_pairs()

    captions = [p[1] for p in pairs]
    tokenizer = build_tokenizer(captions)

    text_encoder = make_text_encoder()
    generator = make_generator()
    discriminator = make_discriminator()
    ema_generator = build_ema_generator(generator)

    gen_optimizer = tf.keras.optimizers.Adam(
        learning_rate=GEN_LR,
        beta_1=0.5,
        beta_2=0.999,
    )

    disc_optimizer = tf.keras.optimizers.Adam(
        learning_rate=DISC_LR,
        beta_1=0.5,
        beta_2=0.999,
    )

    checkpoint = tf.train.Checkpoint(
        generator=generator,
        ema_generator=ema_generator,
        discriminator=discriminator,
        text_encoder=text_encoder,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
    )

    manager = tf.train.CheckpointManager(
        checkpoint,
        str(CHECKPOINT_DIR / "ckpt_60px_conditional_v9"),
        max_to_keep=5,
    )

    start_epoch = 1

    # V9 deliberately does not load V8/V7/V6 checkpoints because the
    # generator now has additional conditioning layers.
    #
    # A V9 checkpoint is resumed only when it belongs to the exact same
    # cleaned caption set.
    current_fingerprint = CAPTION_FINGERPRINT_PATH.read_text(
        encoding="utf-8"
    ).strip()

    checkpoint_fingerprint_path = CHECKPOINT_DIR / "caption_fingerprint.txt"
    can_resume = (
        manager.latest_checkpoint is not None
        and checkpoint_fingerprint_path.exists()
        and checkpoint_fingerprint_path.read_text(
            encoding="utf-8"
        ).strip() == current_fingerprint
    )

    if can_resume:
        print(f"Restoring V9 checkpoint: {manager.latest_checkpoint}")
        checkpoint.restore(manager.latest_checkpoint).expect_partial()

        meta_path = CHECKPOINT_DIR / "last_epoch.txt"
        if meta_path.exists():
            try:
                start_epoch = int(meta_path.read_text().strip()) + 1
            except ValueError:
                start_epoch = 1
    else:
        if manager.latest_checkpoint:
            print(
                "Existing V9 checkpoint does not match the current "
                "caption set; starting V9 from scratch."
            )
        else:
            print("Starting V9 from scratch with the current captions.txt.")

    print()
    print("V9 configuration")
    print("================")
    print(f"Pairs used: {len(pairs)}")
    print(f"Generator parameters: {generator.count_params():,}")
    print(f"Discriminator parameters: {discriminator.count_params():,}")
    print(f"Text encoder parameters: {text_encoder.count_params():,}")
    print("Generator architecture: V8 baseline + identity-initialized multi-stage text conditioning")
    print(f"Mismatch loss weight: {MISMATCH_LOSS_WEIGHT}")
    print("Training schedule: epochs 1-300")
    print()

    dataset = make_dataset(pairs, tokenizer)

    fixed_noise = tf.random.normal(
        shape=(BATCH_SIZE, NOISE_DIM),
        seed=SEED,
    )

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        gen_losses = []
        disc_losses = []

        for step, (real_images, caption_tokens) in enumerate(dataset):
            batch_size = tf.shape(real_images)[0]
            noise = tf.random.normal(
                shape=(batch_size, NOISE_DIM)
            )

            for _ in range(N_CRITIC):
                d_loss = disc_step(
                    real_images,
                    caption_tokens,
                    noise,
                    generator,
                    discriminator,
                    text_encoder,
                    disc_optimizer,
                )

            g_loss = gen_step(
                caption_tokens,
                noise,
                generator,
                discriminator,
                text_encoder,
                gen_optimizer,
            )

            if (step + 1) % LOG_EVERY_STEPS == 0:
                pass

            disc_losses.append(float(d_loss.numpy()))
            gen_losses.append(float(g_loss.numpy()))

        if epoch % EMA_UPDATE_EVERY == 0:
            update_ema(
                ema_generator,
                generator,
                EMA_DECAY,
            )

        gen_mean = float(np.mean(gen_losses))
        disc_mean = float(np.mean(disc_losses))
        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{EPOCHS}  "
            f"Gen {gen_mean:.4f}  "
            f"Disc {disc_mean:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if epoch % SAVE_EVERY_EPOCHS == 0:
            checkpoint_path = manager.save()
            (CHECKPOINT_DIR / "last_epoch.txt").write_text(
                str(epoch),
                encoding="utf-8",
            )
            (CHECKPOINT_DIR / "caption_fingerprint.txt").write_text(
                current_fingerprint,
                encoding="utf-8",
            )
            print(f"Saved checkpoint: {checkpoint_path}")

            save_sample_grid(
                ema_generator,
                text_encoder,
                tokenizer,
                epoch,
            )

            if epoch % 50 == 0:
                save_caption_sweep(
                    ema_generator,
                    text_encoder,
                    tokenizer,
                    epoch,
                )

    generator.save(GENERATOR_PATH)
    ema_generator.save(EMA_GENERATOR_PATH)
    discriminator.save(DISCRIMINATOR_PATH)
    text_encoder.save(TEXT_ENCODER_PATH)

    save_sample_grid(
        ema_generator,
        text_encoder,
        tokenizer,
        EPOCHS,
    )

    save_caption_sweep(
        ema_generator,
        text_encoder,
        tokenizer,
        EPOCHS,
    )

    print()
    print("V9 training complete.")
    print(f"Models saved in: {MODEL_DIR}")
    print(f"Training captions saved in: {DATA_OUT_DIR / 'captions_used_v9.txt'}")


if __name__ == "__main__":
    main()