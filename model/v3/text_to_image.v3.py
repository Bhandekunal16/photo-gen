import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras import layers

# ============================================================
# V2 MODEL CONFIG
# ============================================================

SEED = 42

IMG_SIZE = 60
CHANNELS = 3
NOISE_DIM = 128
MAX_LEN = 20
VOCAB_SIZE = 5000
EMBED_DIM = 128

MODEL_DIR = "./model/v2"

EMA_GENERATOR_PATH = os.path.join(
    MODEL_DIR,
    "generator_ema_model_60px_finetuned.keras",
)

TEXT_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "text_encoder_60px_finetuned.keras",
)

TOKENIZER_PATH = os.path.join(
    MODEL_DIR,
    "tokenizer_60px_finetuned.json",
)

OUTPUT_DIR = os.path.join(MODEL_DIR, "inference")

# These captions are intentionally different so we can test whether
# the generator responds to the text condition.
CAPTIONS = [
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


def configure_runtime():
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass


def check_required_files():
    required = [
        EMA_GENERATOR_PATH,
        TEXT_ENCODER_PATH,
        TOKENIZER_PATH,
    ]

    missing = [path for path in required if not os.path.isfile(path)]

    if missing:
        raise FileNotFoundError(
            "Missing v2 model files:\n" + "\n".join(f"  - {path}" for path in missing)
        )


def load_v2_models():
    check_required_files()

    print(f"Loading EMA generator: {EMA_GENERATOR_PATH}")
    generator = tf.keras.models.load_model(
        EMA_GENERATOR_PATH,
        custom_objects={
            "ConditioningAugmentation": ConditioningAugmentation,
        },
        compile=False,
    )

    print(f"Loading text encoder: {TEXT_ENCODER_PATH}")
    text_encoder = tf.keras.models.load_model(
        TEXT_ENCODER_PATH,
        compile=False,
    )

    print(f"Loading tokenizer: {TOKENIZER_PATH}")
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    return generator, text_encoder, tokenizer


def make_caption_tokens(tokenizer):
    sequences = tokenizer.texts_to_sequences(CAPTIONS)

    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="pre",
        truncating="pre",
    ).astype(np.int32)

    return tf.constant(padded, dtype=tf.int32)


def generate_caption_grid(generator, text_encoder, tokenizer):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CRITICAL:
    # One identical noise vector is used for every caption.
    # Therefore visual changes are much easier to attribute to text.
    noise = tf.random.stateless_normal(
        [1, NOISE_DIM],
        seed=[SEED, 0],
    )

    caption_tokens = make_caption_tokens(tokenizer)
    text_features = text_encoder(caption_tokens, training=False)

    # Repeat the same noise once for every caption.
    noise_batch = tf.repeat(noise, repeats=len(CAPTIONS), axis=0)

    images = generator(
        [noise_batch, text_features],
        training=False,
    )

    images = tf.clip_by_value(
        (images + 1.0) / 2.0,
        0.0,
        1.0,
    )

    # 9 images -> 3x3 grid.
    grid = 3

    tiled = tf.reshape(
        images,
        [grid, grid, IMG_SIZE, IMG_SIZE, CHANNELS],
    )

    tiled = tf.transpose(
        tiled,
        [0, 2, 1, 3, 4],
    )

    tiled = tf.reshape(
        tiled,
        [grid * IMG_SIZE, grid * IMG_SIZE, CHANNELS],
    )

    tiled_u8 = tf.cast(tiled * 255.0, tf.uint8)

    output_path = os.path.join(
        OUTPUT_DIR,
        "caption_conditioning_same_noise.png",
    )

    tf.io.write_file(
        output_path,
        tf.io.encode_png(tiled_u8),
    )

    print()
    print("Caption-conditioning test:")
    print("----------------------------------------")

    for index, caption in enumerate(CAPTIONS, start=1):
        print(f"{index}. {caption}")

    print("----------------------------------------")
    print(f"Same noise used for all {len(CAPTIONS)} captions.")
    print(f"Saved: {output_path}")

    return output_path


def main():
    configure_runtime()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("V2 text-to-image inference")
    print("===========================")

    generator, text_encoder, tokenizer = load_v2_models()

    print(f"Generator parameters: {generator.count_params():,}")
    print(f"Text encoder parameters: {text_encoder.count_params():,}")

    generate_caption_grid(
        generator,
        text_encoder,
        tokenizer,
    )


if __name__ == "__main__":
    main()
