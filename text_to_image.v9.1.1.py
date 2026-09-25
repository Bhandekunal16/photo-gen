"""V9.1 Fine-tune — refine the completed V9.1 model at low learning rates.

This script does NOT retrain V9.1 from scratch. It restores the latest V9.1
model checkpoint, keeps the exact V9.1 cleaned captions/tokenizer, creates
FRESH low-learning-rate optimizers, and fine-tunes the existing weights.

The V9.1 architecture and image-text contrastive objective are unchanged.
The fine-tune outputs are written to ./model/v9_1_finetune so the original
V9.1 experiment remains untouched.
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

EPOCHS = 100

# Fine-tuning starts from the completed V9.1 checkpoint and uses much
# smaller learning rates than the original V9.1 training run.
FINETUNE_GEN_LR = 5e-5
FINETUNE_DISC_LR = 2.5e-5
MAX_LEN = 20
VOCAB_SIZE = 5000
EMBED_DIM = 128
N_CRITIC = 1

GEN_LR = FINETUNE_GEN_LR
DISC_LR = FINETUNE_DISC_LR

MISMATCH_LOSS_WEIGHT = 0.75
ALIGNMENT_LOSS_WEIGHT = 0.50
ALIGNMENT_TEMPERATURE = 0.10

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

SOURCE_V91_MODEL_DIR = "./model/v9_1"
V91_MODEL_DIR = "./model/v9_1_finetune"
V91_CHECKPOINT_DIR = os.path.join(V91_MODEL_DIR, "checkpoints")
V91_DATA_DIR = os.path.join(V91_MODEL_DIR, "data")

GENERATOR_PATH = os.path.join(V91_MODEL_DIR, "generator_model_60px_v91.keras")
EMA_GENERATOR_PATH = os.path.join(V91_MODEL_DIR, "generator_ema_model_60px_v91.keras")
DISCRIMINATOR_PATH = os.path.join(V91_MODEL_DIR, "discriminator_model_60px_v91.keras")
TEXT_ENCODER_PATH = os.path.join(V91_MODEL_DIR, "text_encoder_60px_v91.keras")
TOKENIZER_PATH = os.path.join(V91_MODEL_DIR, "tokenizer_60px_v91.json")
CLEAN_CAPTIONS_PATH = os.path.join(V91_DATA_DIR, "captions_used_v91.txt")
SOURCE_CLEAN_CAPTIONS_PATH = os.path.join(
    SOURCE_V91_MODEL_DIR, "data", "captions_used_v91.txt"
)
SOURCE_TOKENIZER_PATH = os.path.join(
    SOURCE_V91_MODEL_DIR, "tokenizer_60px_v91.json"
)

MONITOR_CAPTIONS = []


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


def prepare_v91_captions():
    if not os.path.isfile(CAPTION_FILE):
        raise FileNotFoundError(f"Caption file not found: {CAPTION_FILE}")

    os.makedirs(V91_DATA_DIR, exist_ok=True)

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
    print("V9.1 caption preparation")
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


def load_v91_training_artifacts():
    """Load the exact caption set and tokenizer used by completed V9.1."""
    if not os.path.isfile(SOURCE_CLEAN_CAPTIONS_PATH):
        raise FileNotFoundError(
            f"V9.1 cleaned captions not found: {SOURCE_CLEAN_CAPTIONS_PATH}"
        )

    if not os.path.isfile(SOURCE_TOKENIZER_PATH):
        raise FileNotFoundError(
            f"V9.1 tokenizer not found: {SOURCE_TOKENIZER_PATH}"
        )

    rows = []
    with open(SOURCE_CLEAN_CAPTIONS_PATH, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or "|" not in line:
                continue
            image_name, caption = line.split("|", 1)
            image_name = image_name.strip()
            caption = caption.strip()
            image_path = os.path.join(IMAGE_DIR, image_name)
            if os.path.isfile(image_path) and caption:
                rows.append((image_name, caption))

    if len(rows) < BATCH_SIZE:
        raise RuntimeError(
            f"Only {len(rows)} usable pairs found in the V9.1 training artifacts."
        )

    tokenizer_json = Path(SOURCE_TOKENIZER_PATH).read_text(encoding="utf-8")
    tokenizer = tokenizer_from_json(tokenizer_json)

    print()
    print("V9.1 fine-tune data")
    print("===================")
    print(f"Pairs loaded:        {len(rows)}")
    print(f"Source captions:     {SOURCE_CLEAN_CAPTIONS_PATH}")
    print(f"Source tokenizer:    {SOURCE_TOKENIZER_PATH}")
    print()

    return rows, tokenizer



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
        shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
        name="image_input",
    )
    text_input = tf.keras.Input(
        shape=(EMBED_DIM,),
        name="text_input",
    )

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
    conditional_score = layers.Add(name="conditional_score")(
        [realness, compatibility]
    )

    return tf.keras.Model(
        [image_input, text_input],
        [conditional_score, image_features, text_features],
        name="conditional_discriminator_v91",
    )



def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


def discriminator_loss(real_output, fake_output):
    return (
        tf.reduce_mean(tf.nn.relu(1.0 - real_output))
        + tf.reduce_mean(tf.nn.relu(1.0 + fake_output))
    )


def image_text_contrastive_loss(
    image_features,
    text_features,
    temperature=ALIGNMENT_TEMPERATURE,
):
    image_features = tf.math.l2_normalize(image_features, axis=1)
    text_features = tf.math.l2_normalize(text_features, axis=1)
    logits = tf.matmul(
        image_features,
        text_features,
        transpose_b=True,
    ) / temperature
    labels = tf.range(tf.shape(logits)[0])

    image_to_text = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits,
        )
    )
    text_to_image = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=tf.transpose(logits),
        )
    )
    return 0.5 * (image_to_text + text_to_image)


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

        real_output, real_image_features, real_text_features = discriminator(
            [images, text_features], training=True
        )
        fake_output, _, _ = discriminator(
            [tf.stop_gradient(fake_images), text_features], training=True
        )

        mismatched_tokens = tf.roll(caption_tokens, shift=1, axis=0)
        mismatched_features = text_encoder(mismatched_tokens, training=True)
        mismatch_output, _, _ = discriminator(
            [images, mismatched_features], training=True
        )

        adv_loss = discriminator_loss(real_output, fake_output)
        mismatch_loss = tf.reduce_mean(tf.nn.relu(1.0 + mismatch_output))
        alignment_loss = image_text_contrastive_loss(
            real_image_features,
            real_text_features,
        )

        total_loss = (
            adv_loss
            + MISMATCH_LOSS_WEIGHT * mismatch_loss
            + ALIGNMENT_LOSS_WEIGHT * alignment_loss
        )

    trainable_vars = discriminator.trainable_variables + text_encoder.trainable_variables
    gradients = tape.gradient(total_loss, trainable_vars)
    disc_opt.apply_gradients(zip(gradients, trainable_vars))
    return total_loss, alignment_loss


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
        text_features = tf.stop_gradient(
            text_encoder(caption_tokens, training=False)
        )
        noise = tf.random.normal([batch_size, NOISE_DIM])
        fake_images = generator([noise, text_features], training=True)

        fake_output, fake_image_features, fake_text_features = discriminator(
            [fake_images, text_features], training=False
        )

        adv_loss = generator_loss(fake_output)
        alignment_loss = image_text_contrastive_loss(
            fake_image_features,
            tf.stop_gradient(fake_text_features),
        )
        total_loss = adv_loss + ALIGNMENT_LOSS_WEIGHT * alignment_loss

    gradients = tape.gradient(total_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(gradients, generator.trainable_variables))
    return total_loss, alignment_loss



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
        V91_MODEL_DIR,
        "gen_images",
        "v9_1",
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


def save_caption_sweep(generator, text_encoder, tokenizer, epoch):
    """Generate all monitor captions from exactly the same noise vector."""
    tokens = tf.constant(
        pad_sequences(
            tokenizer.texts_to_sequences(MONITOR_CAPTIONS),
            maxlen=MAX_LEN,
            padding="pre",
            truncating="pre",
        ).astype(np.int32),
        dtype=tf.int32,
    )

    base_noise = tf.random.stateless_normal(
        [1, NOISE_DIM],
        seed=[SEED, 999],
    )
    noise = tf.repeat(base_noise, MONITOR_GRID, axis=0)
    text_features = text_encoder(tokens, training=False)

    images = generator(
        [noise, text_features],
        training=False,
    )
    images = tf.clip_by_value((images + 1.0) / 2.0, 0.0, 1.0)

    grid = int(np.ceil(np.sqrt(MONITOR_GRID)))
    tiled = tf.reshape(
        images,
        [grid, grid, IMG_SIZE, IMG_SIZE, CHANNELS],
    )
    tiled = tf.transpose(tiled, [0, 2, 1, 3, 4])
    tiled = tf.reshape(
        tiled,
        [grid * IMG_SIZE, grid * IMG_SIZE, CHANNELS],
    )

    out_dir = os.path.join(V91_MODEL_DIR, "gen_images", "v9_1")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"caption_sweep_epoch_{epoch}.png")
    tf.io.write_file(
        out_path,
        tf.io.encode_png(tf.cast(tiled * 255.0, tf.uint8)),
    )

    Path(os.path.join(out_dir, f"caption_sweep_epoch_{epoch}.txt")).write_text(
        "\n".join(f"{i + 1}. {caption}" for i, caption in enumerate(MONITOR_CAPTIONS)),
        encoding="utf-8",
    )
    print(f"Saved caption sweep: {out_path}")


def main():
    configure_runtime()

    os.makedirs(V91_MODEL_DIR, exist_ok=True)
    os.makedirs(V91_CHECKPOINT_DIR, exist_ok=True)

    rows, tokenizer = load_v91_training_artifacts()

    global MONITOR_CAPTIONS
    monitor_indices = np.linspace(
        0,
        len(rows) - 1,
        num=min(MONITOR_GRID, len(rows)),
        dtype=int,
    )
    MONITOR_CAPTIONS = [rows[int(i)][1] for i in monitor_indices]

    payload = load_image_caption_dataset(rows, tokenizer)

    text_encoder = make_text_encoder()
    generator = make_generator()
    discriminator = make_discriminator()

    dummy_tokens = tf.zeros([1, MAX_LEN], dtype=tf.int32)
    dummy_text = text_encoder(dummy_tokens, training=False)
    dummy_noise = tf.zeros([1, NOISE_DIM], dtype=tf.float32)

    generator([dummy_noise, dummy_text], training=False)
    discriminator(
        [
            tf.zeros([1, IMG_SIZE, IMG_SIZE, CHANNELS]),
            dummy_text,
        ],
        training=False,
    )

    g_ema = make_generator()
    g_ema([dummy_noise, dummy_text], training=False)

    # Fresh optimizers are intentional: do not carry the original V9.1
    # optimizer momentum/slot state into the low-LR fine-tuning phase.
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

    source_checkpoint_dir = os.path.join(
        SOURCE_V91_MODEL_DIR,
        "checkpoints",
    )
    latest_ckpt = tf.train.latest_checkpoint(source_checkpoint_dir)
    if not latest_ckpt:
        raise FileNotFoundError(
            "No completed V9.1 checkpoint was found in "
            f"{source_checkpoint_dir}. Finish V9.1 first."
        )

    # Restore ONLY model weights from V9.1. The optimizer state is deliberately
    # excluded so fine-tuning starts with fresh low-LR Adam optimizers.
    restore_checkpoint = tf.train.Checkpoint(
        text_encoder=text_encoder,
        generator=generator,
        discriminator=discriminator,
        g_ema=g_ema,
    )
    restore_checkpoint.restore(latest_ckpt).expect_partial()
    print(f"Restored V9.1 weights: {latest_ckpt}")

    # The EMA generator is already restored from V9.1. Keep it as the primary
    # sampling model during fine-tuning.
    g_ema.trainable = False

    epoch_var = tf.Variable(
        0,
        dtype=tf.int64,
        trainable=False,
        name="v91_finetune_epoch",
    )

    checkpoint_prefix = os.path.join(
        V91_CHECKPOINT_DIR,
        "ckpt_60px_v91_finetune",
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

    # If a fine-tune checkpoint already exists, resume the fine-tune without
    # touching the original V9.1 checkpoint.
    existing_ft = tf.train.latest_checkpoint(V91_CHECKPOINT_DIR)
    if existing_ft:
        checkpoint.restore(existing_ft).expect_partial()
        print(f"Resumed fine-tune checkpoint: {existing_ft}")

    print()
    print("V9.1 fine-tune configuration")
    print("============================")
    print(f"Pairs used: {len(rows)}")
    print(f"Generator parameters: {generator.count_params():,}")
    print(f"Discriminator parameters: {discriminator.count_params():,}")
    print(f"Text encoder parameters: {text_encoder.count_params():,}")
    print("Architecture: unchanged V9.1")
    print(f"Generator LR: {GEN_LR:.2e}")
    print(f"Discriminator LR: {DISC_LR:.2e}")
    print(f"Alignment loss weight: {ALIGNMENT_LOSS_WEIGHT:.2f}")
    print(f"Alignment temperature: {ALIGNMENT_TEMPERATURE:.2f}")
    print(f"Fine-tune epochs: 1-{EPOCHS}")
    print()

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
    monitor_text = text_encoder(monitor_tokens, training=False)

    g_metric = tf.keras.metrics.Mean()
    d_metric = tf.keras.metrics.Mean()
    align_metric = tf.keras.metrics.Mean()

    while int(epoch_var.numpy()) < EPOCHS:
        g_metric.reset_state()
        d_metric.reset_state()
        align_metric.reset_state()

        epoch_start = time.time()
        actual_epoch = int(epoch_var.numpy()) + 1

        for step_idx, (image_batch, caption_batch) in enumerate(payload):
            d_loss, d_alignment = disc_step(
                image_batch,
                caption_batch,
                text_encoder,
                generator,
                discriminator,
                disc_opt,
            )
            d_metric.update_state(d_loss)
            align_metric.update_state(d_alignment)

            if step_idx % N_CRITIC == 0:
                g_loss, g_alignment = gen_step(
                    caption_batch,
                    text_encoder,
                    generator,
                    discriminator,
                    gen_opt,
                )
                g_metric.update_state(g_loss)
                align_metric.update_state(g_alignment)

                if step_idx % EMA_UPDATE_EVERY == 0:
                    update_ema(generator, g_ema, EMA_DECAY)

            if (step_idx + 1) % LOG_EVERY_STEPS == 0:
                sys.stdout.write(
                    f"\rFT Epoch {actual_epoch:>3}  "
                    f"step {step_idx + 1:>5}  "
                    f"g={float(g_metric.result()):.4f}  "
                    f"d={float(d_metric.result()):.4f}  "
                    f"align={float(align_metric.result()):.4f}"
                )
                sys.stdout.flush()

        epoch_var.assign(actual_epoch)
        elapsed = time.time() - epoch_start
        g_avg = float(g_metric.result())
        d_avg = float(d_metric.result())
        a_avg = float(align_metric.result())

        sys.stdout.write("\r" + " " * 120 + "\r")
        print(
            f"FT Epoch {actual_epoch}/{EPOCHS}  "
            f"Gen {g_avg:.4f}  "
            f"Disc {d_avg:.4f}  "
            f"Align {a_avg:.4f}  "
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
            print(f"Saved fine-tune checkpoint: {checkpoint_prefix}")

        if actual_epoch % 25 == 0 or actual_epoch == EPOCHS:
            save_caption_sweep(
                g_ema,
                text_encoder,
                tokenizer,
                actual_epoch,
            )

    generator.save(GENERATOR_PATH)
    g_ema.save(EMA_GENERATOR_PATH)
    discriminator.save(DISCRIMINATOR_PATH)
    text_encoder.save(TEXT_ENCODER_PATH)

    Path(CLEAN_CAPTIONS_PATH).write_text(
        "\n".join(f"{image_name}|{caption}" for image_name, caption in rows) + "\n",
        encoding="utf-8",
    )
    Path(TOKENIZER_PATH).write_text(
        tokenizer.to_json(),
        encoding="utf-8",
    )

    print()
    print("V9.1 fine-tuning complete.")
    print(f"Models saved in: {V91_MODEL_DIR}")
    print(f"Fine-tune captions saved in: {CLEAN_CAPTIONS_PATH}")


if __name__ == "__main__":
    main()