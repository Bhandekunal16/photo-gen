"""V7: FiLM-based multi-stage text conditioning for the 60x60 GAN.

V7 starts from the completed V6 model and replaces the additive StageTextBias
conditioning inside the generator with identity-initialized FiLM conditioning.

FiLM applies learned text-dependent scale and shift at multiple spatial stages:
5x5, 10x10, 20x20, 40x40, and 60x60.

The FiLM layers start as identity transforms so V7 initially stays close to the
V6 generator. The discriminator, text encoder, hinge objective, mismatch loss,
EMA, and CPU configuration are otherwise kept unchanged.
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
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = 60
CHANNELS = 3
BATCH_SIZE = 16
NOISE_DIM = 128

# V7 starts from the completed V6 model.
FINE_TUNE_EPOCHS = 150
START_EPOCH = 800

MAX_LEN = 20
VOCAB_SIZE = 5000
EMBED_DIM = 128
N_CRITIC = 1

GEN_LR = 2e-4
DISC_LR = 1e-4

# Keep the V4 conditioning objective; V6 changes the generator architecture.
MISMATCH_LOSS_WEIGHT = 0.75

EMA_DECAY = 0.995
EMA_UPDATE_EVERY = 5
MONITOR_GRID = 9
LOG_EVERY_STEPS = 20

V6_MODEL_DIR = "./model/v6"
V7_MODEL_DIR = "./model/v7"
V7_CHECKPOINT_DIR = os.path.join(V7_MODEL_DIR, "checkpoints")

V6_GENERATOR_PATH = os.path.join(
    V6_MODEL_DIR, "generator_model_60px_v6.keras"
)
V6_EMA_GENERATOR_PATH = os.path.join(
    V6_MODEL_DIR, "generator_ema_model_60px_v6.keras"
)
V6_TEXT_ENCODER_PATH = os.path.join(
    V6_MODEL_DIR, "text_encoder_60px_v6.keras"
)
V6_DISCRIMINATOR_PATH = os.path.join(
    V6_MODEL_DIR, "discriminator_model_60px_v6.keras"
)
V6_TOKENIZER_PATH = os.path.join(
    V6_MODEL_DIR, "tokenizer_60px_v6.json"
)

V7_GENERATOR_PATH = os.path.join(
    V7_MODEL_DIR, "generator_model_60px_v7.keras"
)
V7_EMA_GENERATOR_PATH = os.path.join(
    V7_MODEL_DIR, "generator_ema_model_60px_v7.keras"
)
V7_TEXT_ENCODER_PATH = os.path.join(
    V7_MODEL_DIR, "text_encoder_60px_v7.keras"
)
V7_DISCRIMINATOR_PATH = os.path.join(
    V7_MODEL_DIR, "discriminator_model_60px_v7.keras"
)
V7_TOKENIZER_PATH = os.path.join(
    V7_MODEL_DIR, "tokenizer_60px_v7.json"
)

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


class StageTextBias(layers.Layer):
    """Inject text into a feature map without changing its channel count."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.projection = layers.Dense(
            channels,
            use_bias=True,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="text_projection",
        )

    def build(self, input_shape):
        # input_shape is [feature_shape, text_shape]. Explicitly building the
        # Dense layer removes the Keras unbuilt-state warning during loading.
        text_shape = input_shape[1]
        self.projection.build(text_shape)
        super().build(input_shape)

    def call(self, inputs):
        features, text = inputs
        bias = self.projection(text)
        bias = tf.reshape(bias, [tf.shape(bias)[0], 1, 1, self.channels])
        return features + bias

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


class StageTextFiLM(layers.Layer):
    """
    Identity-initialized FiLM conditioning.

    The text encoder produces per-channel scale and shift parameters.
    At initialization:
        gamma = 1
        beta  = 0
    so the layer initially behaves like an identity transform.
    """

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.projection = layers.Dense(
            channels * 2,
            use_bias=True,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="film_projection",
        )

    def build(self, input_shape):
        text_shape = input_shape[1]
        self.projection.build(text_shape)
        super().build(input_shape)

    def call(self, inputs):
        features, text = inputs

        params = self.projection(text)
        delta_gamma, beta = tf.split(params, 2, axis=-1)

        gamma = 1.0 + delta_gamma

        delta_gamma = tf.reshape(
            gamma,
            [tf.shape(gamma)[0], 1, 1, self.channels],
        )
        beta = tf.reshape(
            beta,
            [tf.shape(beta)[0], 1, 1, self.channels],
        )

        return features * delta_gamma + beta

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


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


def load_image_caption_dataset(img_folder, caption_file, tokenizer):
    image_paths = []
    captions = []

    with open(caption_file, "r", encoding="utf-8") as f:
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


def make_generator_v7():
    """Generator with identity-initialized FiLM text conditioning at every stage."""
    noise_input = tf.keras.Input(shape=(NOISE_DIM,), name="noise_input")
    text_input = tf.keras.Input(shape=(EMBED_DIM,), name="text_input")

    ca = ConditioningAugmentation(
        EMBED_DIM,
        name="conditioning_augmentation",
    )(text_input)

    x = layers.Concatenate(name="noise_text_concat")([noise_input, ca])

    # 5x5
    x = layers.Dense(
        5 * 5 * 128,
        use_bias=False,
        name="base_dense",
    )(x)
    x = layers.BatchNormalization(name="base_bn")(x)
    x = layers.LeakyReLU(0.2, name="base_activation")(x)
    x = layers.Reshape((5, 5, 128), name="base_reshape")(x)
    x = StageTextFiLM(128, name="text_film_5x5")([x, text_input])

    # 10x10
    x = layers.UpSampling2D(name="upsample_10x10")(x)
    x = layers.Conv2D(
        64, 3, padding="same", use_bias=False, name="conv_64"
    )(x)
    x = layers.BatchNormalization(name="bn_64")(x)
    x = layers.LeakyReLU(0.2, name="act_64")(x)
    x = StageTextFiLM(64, name="text_film_10x10")([x, text_input])

    # 20x20
    x = layers.UpSampling2D(name="upsample_20x20")(x)
    x = layers.Conv2D(
        32, 3, padding="same", use_bias=False, name="conv_32"
    )(x)
    x = layers.BatchNormalization(name="bn_32")(x)
    x = layers.LeakyReLU(0.2, name="act_32")(x)
    x = StageTextFiLM(32, name="text_film_20x20")([x, text_input])

    # 40x40
    x = layers.UpSampling2D(name="upsample_40x40")(x)
    x = layers.Conv2D(
        16, 3, padding="same", use_bias=False, name="conv_16"
    )(x)
    x = layers.BatchNormalization(name="bn_16")(x)
    x = layers.LeakyReLU(0.2, name="act_16")(x)
    x = StageTextFiLM(16, name="text_film_40x40")([x, text_input])

    # 60x60
    x = layers.Resizing(
        IMG_SIZE,
        IMG_SIZE,
        name="resize_60x60",
    )(x)
    x = layers.Conv2D(
        8, 3, padding="same", use_bias=False, name="conv_8"
    )(x)
    x = layers.BatchNormalization(name="bn_8")(x)
    x = layers.LeakyReLU(0.2, name="act_8")(x)
    x = StageTextFiLM(8, name="text_film_60x60")([x, text_input])

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
        name="generator_v7",
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
        lambda: tf.roll(caption_tokens, shift=1, axis=0),
        lambda: caption_tokens,
    )


def _find_compatible_source_layer(source_model, target_layer):
    """Find a source layer with the same class and exact weight shapes."""
    target_weights = target_layer.get_weights()
    if not target_weights:
        return None

    target_shapes = [tuple(w.shape) for w in target_weights]

    for source_layer in source_model.layers:
        if source_layer.__class__ is not target_layer.__class__:
            continue

        source_weights = source_layer.get_weights()
        if len(source_weights) != len(target_weights):
            continue

        source_shapes = [tuple(w.shape) for w in source_weights]
        if source_shapes == target_shapes:
            return source_layer

    return None


def transfer_compatible_weights(source_model, target_model):
    """
    Transfer V6 generator weights into V7 where the layers are compatible.

    StageTextFiLM layers are intentionally left identity-initialized.
    """
    transferred = []
    used_source_names = set()

    for target_layer in target_model.layers:
        if isinstance(target_layer, StageTextFiLM):
            continue

        if isinstance(
            target_layer,
            (
                layers.InputLayer,
                layers.Reshape,
                layers.UpSampling2D,
                layers.Resizing,
                layers.Concatenate,
                layers.LeakyReLU,
            ),
        ):
            continue

        source_layer = _find_compatible_source_layer(
            source_model,
            target_layer,
        )

        if source_layer is None:
            continue

        if source_layer.name in used_source_names:
            continue

        target_layer.set_weights(source_layer.get_weights())
        used_source_names.add(source_layer.name)
        transferred.append((source_layer.name, target_layer.name))

    print(
        f"Transferred {len(transferred)} compatible V6 generator layers into V7."
    )

    for source_name, target_name in transferred:
        print(f"  {source_name} -> {target_name}")

    film_count = sum(
        isinstance(layer, StageTextFiLM)
        for layer in target_model.layers
    )

    print(
        f"Identity-initialized FiLM layers: {film_count}"
    )


def build_ema_generator(generator):
    ema = make_generator_v7()
    # Build before copying weights.
    dummy_noise = tf.zeros([1, NOISE_DIM])
    dummy_text = tf.zeros([1, EMBED_DIM])
    ema([dummy_noise, dummy_text], training=False)
    ema.set_weights(generator.get_weights())
    return ema


def update_ema(generator, ema, decay):
    for ema_var, var in zip(ema.weights, generator.weights):
        ema_var.assign(decay * ema_var + (1.0 - decay) * var)


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
        fake_output = discriminator(
            [fake_images_for_d, text_features], training=True
        )

        mismatched_tokens = make_mismatched_captions(caption_tokens)
        mismatched_features = text_encoder(mismatched_tokens, training=True)
        mismatch_output = discriminator(
            [images, mismatched_features], training=True
        )

        d_loss = discriminator_loss(real_output, fake_output)
        mismatch_loss = tf.reduce_mean(tf.nn.relu(1.0 + mismatch_output))
        d_loss = d_loss + MISMATCH_LOSS_WEIGHT * mismatch_loss

    trainable_vars = discriminator.trainable_variables + text_encoder.trainable_variables
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
        text_features = text_encoder(caption_tokens, training=True)
        noise = tf.random.normal([batch_size, NOISE_DIM])
        fake_images = generator([noise, text_features], training=True)
        fake_output = discriminator(
            [fake_images, text_features], training=True
        )
        g_loss = generator_loss(fake_output)

    trainable_vars = generator.trainable_variables + text_encoder.trainable_variables
    grads = tape.gradient(g_loss, trainable_vars)
    gen_opt.apply_gradients(zip(grads, trainable_vars))

    return g_loss


def _resolve_captions_path():
    candidates = [
        "./data/captions_60px_v2.txt",
        "./data/captions.txt",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No captions file found. Expected one of:\n" +
        "\n".join(f"  - {p}" for p in candidates)
    )


def save_generated_samples(epoch_number, folder_type, generator, monitor_noise, monitor_text):
    images = generator([monitor_noise, monitor_text], training=False)
    images = tf.clip_by_value((images + 1.0) / 2.0, 0.0, 1.0)

    grid = int(np.ceil(np.sqrt(MONITOR_GRID)))
    images = tf.reshape(
        images,
        [grid, grid, IMG_SIZE, IMG_SIZE, CHANNELS],
    )
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.reshape(
        images,
        [grid * IMG_SIZE, grid * IMG_SIZE, CHANNELS],
    )

    out_dir = os.path.join(V7_MODEL_DIR, "gen_images", folder_type)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        f"generated_image_epoch_{epoch_number}.png",
    )
    tf.io.write_file(out_path, tf.io.encode_png(tf.cast(images * 255.0, tf.uint8)))
    print(f"Saved samples: {out_path}")


def main():
    configure_runtime()

    os.makedirs(V7_MODEL_DIR, exist_ok=True)
    os.makedirs(V7_CHECKPOINT_DIR, exist_ok=True)

    captions_path = _resolve_captions_path()
    print(f"Using captions file: {captions_path}")

    if not os.path.isfile(V6_TOKENIZER_PATH):
        raise FileNotFoundError(
            f"V6 tokenizer not found: {V6_TOKENIZER_PATH}"
        )

    with open(V6_TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    payload = load_image_caption_dataset(
        "./data/image60px",
        captions_path,
        tokenizer,
    )

    required_v6 = [
        V6_GENERATOR_PATH,
        V6_EMA_GENERATOR_PATH,
        V6_TEXT_ENCODER_PATH,
        V6_DISCRIMINATOR_PATH,
    ]

    missing = [path for path in required_v6 if not os.path.isfile(path)]

    if missing:
        raise FileNotFoundError(
            "V6 models are required to start V7:\n" +
            "\n".join(f"  - {path}" for path in missing)
        )

    print("Loading V6 models...")

    v6_generator = tf.keras.models.load_model(
        V6_GENERATOR_PATH,
        custom_objects={
            "ConditioningAugmentation": ConditioningAugmentation,
            "StageTextBias": StageTextBias,
        },
        compile=False,
    )

    v6_ema = tf.keras.models.load_model(
        V6_EMA_GENERATOR_PATH,
        custom_objects={
            "ConditioningAugmentation": ConditioningAugmentation,
            "StageTextBias": StageTextBias,
        },
        compile=False,
    )

    text_encoder = tf.keras.models.load_model(
        V6_TEXT_ENCODER_PATH,
        compile=False,
    )

    discriminator = tf.keras.models.load_model(
        V6_DISCRIMINATOR_PATH,
        compile=False,
    )

    generator = make_generator_v7()

    generator(
        [
            tf.zeros([1, NOISE_DIM]),
            tf.zeros([1, EMBED_DIM]),
        ],
        training=False,
    )

    transfer_compatible_weights(
        v6_generator,
        generator,
    )

    v7_ema = make_generator_v7()

    v7_ema(
        [
            tf.zeros([1, NOISE_DIM]),
            tf.zeros([1, EMBED_DIM]),
        ],
        training=False,
    )

    transfer_compatible_weights(
        v6_ema,
        v7_ema,
    )

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
        START_EPOCH,
        dtype=tf.int64,
        trainable=False,
        name="v7_epoch",
    )

    target_epoch = START_EPOCH + FINE_TUNE_EPOCHS

    checkpoint_prefix = os.path.join(
        V7_CHECKPOINT_DIR,
        "ckpt_60px_conditional_v7",
    )

    checkpoint = tf.train.Checkpoint(
        epoch=epoch_var,
        text_encoder=text_encoder,
        generator=generator,
        discriminator=discriminator,
        g_ema=v7_ema,
        g_optimizer=gen_opt,
        d_optimizer=disc_opt,
    )

    latest_ckpt = tf.train.latest_checkpoint(
        V7_CHECKPOINT_DIR
    )

    if latest_ckpt:
        checkpoint.restore(latest_ckpt).expect_partial()
        print(f"Resumed V7 checkpoint: {latest_ckpt}")
    else:
        print("Starting V7 from the saved V6 models.")

    gen_opt.learning_rate.assign(GEN_LR)
    disc_opt.learning_rate.assign(DISC_LR)

    print()
    print("V7 configuration")
    print("=================")
    print(f"Generator parameters: {generator.count_params():,}")
    print(f"Discriminator parameters: {discriminator.count_params():,}")
    print(f"Text encoder parameters: {text_encoder.count_params():,}")
    print("Generator conditioning: FiLM at 5x5, 10x10, 20x20, 40x40, 60x60")
    print("FiLM initialization: identity (gamma=1, beta=0)")
    print(f"Mismatch loss weight: {MISMATCH_LOSS_WEIGHT:.2f}")
    print(
        f"Training schedule: epochs {int(epoch_var.numpy()) + 1}-"
        f"{target_epoch} | "
        f"gen_lr={GEN_LR:.1e} | disc_lr={DISC_LR:.1e}"
    )

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

    while int(epoch_var.numpy()) < target_epoch:
        g_metric.reset_state()
        d_metric.reset_state()

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
                        v7_ema,
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

        sys.stdout.write("\r" + " " * 90 + "\r")

        print(
            f"Epoch {actual_epoch}/{target_epoch}  "
            f"Gen {g_avg:.4f}  "
            f"Disc {d_avg:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if actual_epoch % 10 == 0 or actual_epoch == target_epoch:
            save_generated_samples(
                actual_epoch,
                "v7",
                v7_ema,
                monitor_noise,
                monitor_text,
            )

        if actual_epoch % 10 == 0:
            checkpoint.save(
                file_prefix=checkpoint_prefix
            )

    generator.save(V7_GENERATOR_PATH)
    v7_ema.save(V7_EMA_GENERATOR_PATH)
    discriminator.save(V7_DISCRIMINATOR_PATH)
    text_encoder.save(V7_TEXT_ENCODER_PATH)

    with open(
        V7_TOKENIZER_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(tokenizer.to_json())

    print()
    print("V7 training complete.")
    print(f"Models saved in: {V7_MODEL_DIR}")


if __name__ == "__main__":
    main()