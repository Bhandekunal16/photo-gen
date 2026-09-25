# Text-to-Image GAN --- Development Roadmap and Experiment History

## 1. Project Goal

Build a small conditional text-to-image GAN that generates **60×60 RGB
images from captions**, using TensorFlow/Keras and CPU-only training.

The project is intentionally lightweight because the available machine
does not have a GPU.

------------------------------------------------------------------------

# 2. Current Base Configuration

The main training configuration evolved to:

``` python
IMG_SIZE = 60
CHANNELS = 3

BATCH_SIZE = 16
NOISE_DIM = 128

MAX_LEN = 20
VOCAB_SIZE = 5000
EMBED_DIM = 128

N_CRITIC = 1

EMA_DECAY = 0.995
EMA_UPDATE_EVERY = 5

USE_MIXED_PRECISION = False
USE_XLA = False

CPU_INTRA_OP_THREADS = 4
CPU_INTER_OP_THREADS = 2
```

Training uses:

-   TensorFlow/Keras
-   Hinge GAN objective
-   Trainable text encoder
-   Conditioning Augmentation
-   Projection-based conditional discriminator
-   EMA generator
-   CPU-oriented model sizes
-   `tf.data` caching/shuffling/batching/prefetching

------------------------------------------------------------------------

# 3. Dataset and Caption Generation

## Step 1 --- Collect images

The current dataset contains approximately:

``` text
1,000 images
```

The images are resized to:

``` text
60 × 60 × 3
```

The relatively small number of images is an important limitation.

------------------------------------------------------------------------

## Step 2 --- Generate captions with BLIP

The caption-generation pipeline uses:

``` text
Salesforce/blip-image-captioning-base
```

The basic process is:

``` text
image
  ↓
BLIP processor
  ↓
BLIP model
  ↓
caption
  ↓
image.jpg|caption
```

The caption file format is:

``` text
image_001.jpg|a caption describing the image
image_002.jpg|another caption
image_003.jpg|another caption
```

The current caption-generation script generates **one caption per
image**.

Important:

``` python
with open(caption_file, "w")
```

means the caption file is overwritten when the script is run.

------------------------------------------------------------------------

# 4. Important Caption-Quality Finding

During dataset inspection, several captions were found to be
contradictory or inaccurate.

For example, an image could have descriptions referring to different
subjects.

This means:

``` text
image
  ↓
incorrect/noisy caption
```

can provide conflicting supervision to the GAN.

This is one of the major limitations of the current dataset.

Therefore, caption quality must be considered when evaluating
text-to-image performance.

------------------------------------------------------------------------

# 5. V1 --- Initial Conditional GAN

The first model used:

``` text
noise
  +
text condition
  ↓
generator
  ↓
image
```

and a conditional discriminator.

The initial implementation had several problems:

-   Text representation was effectively not trained with the GAN.
-   Discriminator conditioning was relatively weak.
-   Gradient penalty code existed but was not actually used.
-   The GAN objective was therefore effectively hinge GAN rather than
    WGAN-GP.
-   The loss values were incorrectly considered as a possible "perfect
    image" criterion.
-   The discriminator used relatively large channel sizes for CPU
    training.

These issues became the basis for the next version.

------------------------------------------------------------------------

# 6. V2 --- Main Stable Baseline

V2 fixed the major architectural/training issues.

## 6.1 Trainable text encoder

The text encoder became:

``` text
Token IDs
   ↓
Embedding
   ↓
LSTM
   ↓
Layer Normalization
   ↓
Text embedding
```

This encoder is trained together with the GAN.

The text encoder is also saved separately.

------------------------------------------------------------------------

## 6.2 Conditioning Augmentation

The generator uses a conditioning-augmentation layer.

Conceptually:

``` text
text embedding
      ↓
mean + log variance
      ↓
sampled conditioning vector
      ↓
generator
```

This gives the generator a smoother text-conditioned latent
representation.

------------------------------------------------------------------------

## 6.3 Projection discriminator

The discriminator was changed from simple concatenation to a
projection-style conditional discriminator.

Conceptually:

``` text
image
 ↓
CNN
 ↓
image feature ──────┐
                    │ compatibility
caption             │
 ↓                  │
text encoder ───────┘
```

The discriminator produces:

``` text
realness score
+
image/text compatibility
```

------------------------------------------------------------------------

## 6.4 Mismatched-caption training

The discriminator also sees:

``` text
real image + correct caption     → positive
real image + wrong caption       → negative
fake image + caption             → negative
```

The wrong caption is created by rotating captions within the batch.

This explicitly teaches the discriminator that the image and caption
must correspond.

------------------------------------------------------------------------

## 6.5 Hinge GAN

The model uses a hinge GAN objective.

`N_CRITIC = 1` is used.

The old unused gradient-penalty implementation was removed.

------------------------------------------------------------------------

## 6.6 EMA generator

An exponential moving average generator is maintained:

``` python
EMA_DECAY = 0.995
```

The EMA generator is preferred for image generation/inference because it
generally provides a smoother version of the trained generator.

------------------------------------------------------------------------

# 7. CPU Optimization

Because there is no GPU, the architecture was reduced.

The CPU-oriented configuration uses:

``` text
Batch size: 16
Noise dimension: 128
Text embedding: 128
Image size: 60×60
```

Generator channels were reduced approximately to:

``` text
128 → 64 → 32 → 16 → 8
```

Discriminator channels were reduced approximately to:

``` text
32 → 64 → 128 → 128
```

The following remain disabled:

``` text
Mixed precision = False
XLA = False
```

oneDNN CPU optimizations remain enabled.

------------------------------------------------------------------------

# 8. V2 Training Result

The first major training run was completed to:

``` text
Epoch 300
```

The generated images progressed from diffuse gray/blue textures toward
coarse landscape-like structures.

Early output:

``` text
mostly blurry texture/noise
```

Later output:

``` text
sky-like region
horizon
land/water-like regions
stronger colors
coarse scene structure
```

This showed that the GAN had learned a useful visual distribution.

However, the output was still mostly generic landscape imagery.

------------------------------------------------------------------------

# 9. V2 Fine-Tuning to Epoch 500

The epoch-300 V2 checkpoint was fine-tuned.

The discriminator learning rate was reduced:

``` python
GEN_LR = 2e-4
DISC_LR = 1e-4
```

The purpose was to reduce discriminator dominance.

The fine-tuned run reached:

``` text
Epoch 500
```

The images became more structured and textured.

This became the main V2 epoch-500 baseline.

------------------------------------------------------------------------

# 10. V3 --- Caption Conditioning Evaluation

V3 was created as an **inference/evaluation script**, not another
training experiment.

The goal was to determine:

> Does the generated image actually respond to the caption?

The test used:

``` text
same noise vector
+
different captions
```

Example captions:

``` text
a body of water
a mountain landscape
a green field
a sunset over the ocean
a city skyline
a forest with trees
a road through the mountains
a beach with waves
a lake with mountains in the background
```

Using the same noise is important because it reduces the influence of
random latent variation.

------------------------------------------------------------------------

# 11. V3 Result

The V3 test showed:

-   Different captions produced different outputs.
-   Green-field/forest captions tended to produce greener outputs.
-   Water/beach captions produced water/horizon-like structures.
-   Sunset captions produced warmer colors.
-   Some caption influence was clearly present.

However:

``` text
city skyline
road through mountains
forest
beach
```

were not rendered as strongly recognizable semantic objects/scenes.

Conclusion:

``` text
Text → image influence       = learned
Broad semantic conditioning  = partially learned
Specific semantic control    = weak
```

------------------------------------------------------------------------

# 12. V4 --- Stronger Caption Mismatch Training

V4 was designed to strengthen text-image alignment without changing the
basic architecture.

The mismatch loss weight changed from:

``` python
0.5
```

to:

``` python
0.75
```

The idea was to give the discriminator more pressure to reject:

``` text
real image + wrong caption
```

while accepting:

``` text
real image + correct caption
```

The learning rates remained:

``` text
Generator = 2e-4
Discriminator = 1e-4
```

V4 started from the V2 epoch-500 learned models.

------------------------------------------------------------------------

# 13. V4 Result

V4 reached approximately:

``` text
Epoch 650
```

The result showed:

-   Better texture
-   Stronger edges
-   More visual detail
-   More landscape structure

But the output remained mostly:

``` text
generic landscape
```

rather than strongly caption-specific imagery.

The mismatch experiment therefore improved visual structure but did not
clearly solve semantic conditioning.

V4 became an important baseline.

------------------------------------------------------------------------

# 14. V5 --- Explicit Image/Text Alignment Loss

V5 added a direct image/text alignment objective.

Conceptually:

``` text
caption
   ↓
text encoder
   ↓
text projection
        ↘
          similarity
        ↗
image projection
   ↑
generated image
```

The alignment objective encouraged:

``` text
generated image ↔ correct caption
```

and discouraged:

``` text
generated image ↔ other captions
```

This was implemented as a lightweight contrastive/InfoNCE-style
alignment objective rather than adding a large external CLIP model.

The goal was to keep CPU training practical.

------------------------------------------------------------------------

# 15. V5 Result

The alignment loss decreased significantly during early V5 training.

This demonstrated:

``` text
alignment objective = active and trainable
```

However, the visual results showed that the model still tended toward
generic landscape imagery.

The model became more textured and visually structured, but
caption-specific semantics did not improve enough.

The best visual region appeared around approximately:

``` text
Epoch 740–760
```

rather than automatically at the final epoch.

Conclusion:

> Adding another alignment loss alone was not sufficient to solve the
> semantic-conditioning problem.

------------------------------------------------------------------------

# 16. Why We Move to V6

At this point the main problem is no longer:

``` text
Can the GAN generate images?
```

It can.

The main problem is:

``` text
Can the caption continue influencing the image
throughout the generation process?
```

The existing generator primarily introduces the text condition near the
beginning of generation.

As the feature map is repeatedly upsampled and transformed, the original
text information can become weaker.

Therefore V6 changes the **generator architecture** rather than simply
adding another loss.

------------------------------------------------------------------------

# 17. V6 --- Multi-Stage Text Conditioning

V6 introduces text conditioning at multiple generator stages.

Conceptually:

``` text
                         text
                          │
              ┌───────────┼───────────┐
              ↓           ↓           ↓
             5×5         10×10       20×20
              │           │           │
          text inject  text inject text inject
              │           │           │
              └───────────┴───────────┘
                          ↓
                        40×40
                          ↓
                     text inject
                          ↓
                        60×60
                          ↓
                        image
```

The purpose is to keep the caption representation available at different
spatial scales.

------------------------------------------------------------------------

# 18. V6 Initialization Strategy

V6 should not randomly destroy the useful V4 generator.

The new text-conditioning layers are initialized conservatively/at zero
so that the initial V6 generator remains close to the V4 generator.

Conceptually:

``` text
V4 generator
     ↓
V6 initialization
     ↓
V6 output initially ≈ V4 output
     ↓
new text-conditioning layers learn
```

This makes V6 a controlled architectural experiment.

------------------------------------------------------------------------

# 19. V6 Starting Point

V6 starts from:

``` text
V4 epoch 650
```

rather than V5.

Reason:

V5 changed the optimization objective and did not clearly improve
semantic control.

Using V4 gives a cleaner baseline:

``` text
V4 learned generator
      +
new multi-stage text conditioning
      =
V6 experiment
```

------------------------------------------------------------------------

# 20. V6 Training Configuration

The intended V6 configuration remains CPU-friendly:

``` text
Image size:       60×60
Batch size:       16
Noise dimension:  128
Text embedding:   128

Generator LR:     2e-4
Discriminator LR: 1e-4

Mismatch weight:  0.75

EMA decay:        0.995
N_CRITIC:         1

Mixed precision:  disabled
XLA:              disabled
```

The main architectural change is:

``` text
multi-stage text injection
```

not another major loss change.

------------------------------------------------------------------------

# 21. V6 File Organization

The versions should remain separated.

``` text
model/
├── v1/
├── v2/
│   ├── generator_model_60px_finetuned.keras
│   ├── generator_ema_model_60px_finetuned.keras
│   ├── discriminator_model_60px_finetuned.keras
│   ├── text_encoder_60px_finetuned.keras
│   └── tokenizer_60px_finetuned.json
│
├── v3/
│   └── inference/evaluation scripts
│
├── v4/
│   ├── generator_model_60px_v4.keras
│   ├── generator_ema_model_60px_v4.keras
│   ├── discriminator_model_60px_v4.keras
│   └── text_encoder_60px_v4.keras
│
├── v5/
│   ├── generator_model_60px_v5.keras
│   ├── generator_ema_model_60px_v5.keras
│   ├── discriminator_model_60px_v5.keras
│   └── text_encoder_60px_v5.keras
│
└── v6/
    ├── checkpoints/
    ├── gen_images/
    │   └── v6/
    ├── generator_model_60px_v6.keras
    ├── generator_ema_model_60px_v6.keras
    ├── discriminator_model_60px_v6.keras
    ├── text_encoder_60px_v6.keras
    └── tokenizer_60px_v6.json
```

------------------------------------------------------------------------

# 22. What Files Are Needed for Continued Learning?

For a saved model version, the main learned artifacts are:

``` text
generator_model_*.keras
generator_ema_model_*.keras
discriminator_model_*.keras
text_encoder_*.keras
tokenizer_*.json
```

The tokenizer is not a neural network, but it is required because it
defines how captions are converted into token IDs.

Old TensorFlow checkpoint files are useful as backup/recovery because
they can contain optimizer/training state.

Do not delete old checkpoints until the newer version has been
validated.

------------------------------------------------------------------------

# 23. How We Evaluate Each Version

Do not evaluate GAN quality using generator/discriminator loss alone.

For every important version, use:

## A. Fixed-noise caption test

Use the same noise and captions:

``` text
a body of water
a mountain landscape
a green field
a sunset over the ocean
a city skyline
a forest with trees
a road through the mountains
a beach with waves
a lake with mountains in the background
```

This tests:

``` text
caption → generated image
```

------------------------------------------------------------------------

## B. Compare multiple checkpoints

Do not assume the final epoch is best.

Compare:

``` text
epoch 660
epoch 700
epoch 750
epoch 800
```

or equivalent checkpoints.

------------------------------------------------------------------------

## C. Evaluate semantic differences

Look for:

``` text
water → water-like structure
mountain → mountain-like structure
forest → vegetation/tree structure
city → buildings/urban structure
road → road/path structure
beach → beach/water/wave structure
```

The important metric is not merely sharpness.

------------------------------------------------------------------------

# 24. Current Findings

The experiments have established:

``` text
V1
↓
basic GAN

V2
↓
stable conditional GAN
↓
coarse visual structure learned

V3
↓
caption influence confirmed

V4
↓
stronger mismatch conditioning
↓
more texture, but generic landscapes remain

V5
↓
explicit alignment loss
↓
alignment loss improves
↓
semantic improvement still limited

V6
↓
multi-stage text conditioning
↓
current experiment
```

------------------------------------------------------------------------

# 25. V6 Success Criteria

V6 should be considered successful if it produces stronger evidence that
different captions cause different semantic structures.

For example:

``` text
"city skyline"
       ↓
recognizable urban/building structure

"forest with trees"
       ↓
recognizable vegetation/tree structure

"road through the mountains"
       ↓
recognizable road + mountain composition

"beach with waves"
       ↓
recognizable water/beach structure
```

The goal is **not merely sharper or more colorful images**.

The primary V6 question is:

> Does multi-stage text conditioning improve semantic control?

------------------------------------------------------------------------

# 26. V6 Experiment Procedure

Run:

``` bash
python3 text_to_image_v6.py
```

Then monitor the generated samples at regular checkpoints.

Recommended checkpoints:

``` text
V6 epoch 660
V6 epoch 700
V6 epoch 750
V6 epoch 800
```

At each checkpoint, save the generated grid.

After training, run the same fixed-noise caption-conditioning evaluation
used for V3/V4/V5.

Then compare:

``` text
V4 epoch 650
        vs
V6 epoch 660
V6 epoch 700
V6 epoch 750
V6 epoch 800
```

------------------------------------------------------------------------

# 27. Decision After V6

If V6 improves semantic conditioning:

``` text
V6
 ↓
keep architecture
 ↓
fine-tune / improve dataset
```

If V6 improves image quality but not semantic conditioning:

``` text
V6
 ↓
dataset/caption quality becomes the main bottleneck
```

If V6 does not improve either:

``` text
V6
 ↓
reconsider generator/discriminator architecture
```

Do not automatically move to another version without evaluating the
result.

------------------------------------------------------------------------

# 28. Overall Strategy

The development strategy is:

``` text
Fix fundamental problems
        ↓
Reduce CPU cost
        ↓
Train baseline
        ↓
Evaluate captions
        ↓
Change one major factor
        ↓
Compare against baseline
        ↓
Keep or reject the experiment
        ↓
Repeat
```

The most important principle is:

> **Change one major thing per version so we know why the model improved
> or failed.**

Current version:

``` text
V6 = multi-stage text conditioning
```

Current baseline:

``` text
V4 epoch 650
```

Current primary evaluation:

``` text
same noise + different captions
```

Current objective:

``` text
Improve caption → image semantic control
```

------------------------------------------------------------------------

# 29. V7 --- FiLM Multi-Stage Text Conditioning

V7 was the next architectural experiment after V6.

The goal was to replace the additive multi-stage text conditioning with
**FiLM (Feature-wise Linear Modulation)**.

Instead of:

``` text
feature map + text bias
```

V7 uses:

``` text
feature map × text-dependent scale
             +
             text-dependent shift
```

FiLM conditioning was applied at:

``` text
5×5
10×10
20×20
40×40
60×60
```

The FiLM layers were identity-initialized:

``` text
gamma = 1
beta  = 0
```

so the V7 generator initially remained close to the V6 generator.

------------------------------------------------------------------------

# 30. V7 Initialization

V7 successfully loaded the V6 models.

The learned V6 generator contained:

``` text
12 compatible layers
```

that were transferred into V7.

V7 added:

``` text
5 FiLM conditioning layers
```

which were identity-initialized.

Therefore the experiment was:

``` text
V6 learned generator
        +
5 new FiLM conditioning stages
        ↓
V7
```

rather than a complete restart.

------------------------------------------------------------------------

# 31. V7 Training

V7 trained from:

``` text
Epoch 801
```

through:

``` text
Epoch 950
```

Configuration:

``` text
Generator LR      = 2e-4
Discriminator LR  = 1e-4
Mismatch weight   = 0.75
EMA decay         = 0.995
```

The training remained numerically stable.

The final epoch was:

``` text
Epoch 950
Gen  = 1.3739
Disc = 1.1317
```

The loss values did not show an obvious GAN collapse.

------------------------------------------------------------------------

# 32. V7 Visual Evaluation

The generated images were inspected at:

``` text
810
820
830
840
850
860
870
880
890
900
910
920
930
940
950
```

The visual progression showed an important pattern.

## Epoch 810--830

The images still contained:

-   sky/ground separation
-   landscape-like structure
-   multiple colors
-   visible spatial variation
-   some useful coarse image structure

Epoch 810 was one of the strongest V7 visual checkpoints.

------------------------------------------------------------------------

## Epoch 840--880

Image structure began to degrade.

The outputs became increasingly:

-   blurry
-   abstract
-   dominated by large color regions
-   less spatially detailed

The model continued training without an obvious loss explosion, but the
images were getting worse.

------------------------------------------------------------------------

## Epoch 900--950

The degradation became much more obvious.

The generated images contained:

-   broad soft regions
-   reduced spatial detail
-   less recognizable structure
-   increasingly abstract compositions

Therefore the final epoch was **not automatically the best V7 model**.

------------------------------------------------------------------------

# 33. V7 Important Finding

V7 demonstrated an important distinction:

``` text
Training loss stability
        ≠
Image quality
```

The losses remained relatively stable through epoch 950, while the
visual outputs progressively lost useful spatial detail.

Therefore:

> GAN generator/discriminator losses must not be used as the sole
> model-selection criterion.

Fixed-noise generated images remain an important evaluation tool.

------------------------------------------------------------------------

# 34. V7 Conclusion

V7 did not provide sufficient evidence that FiLM multi-stage
conditioning improved the final text-to-image result.

The experiment showed:

``` text
FiLM conditioning
        ↓
caption influence remains present
        ↓
but image structure progressively degrades
```

The visual degradation means V7 should **not automatically become the
new baseline**.

The more useful V7 checkpoints to preserve are approximately:

``` text
V7 epoch 810
V7 epoch 820
V7 epoch 830
```

rather than assuming epoch 950 is best.

------------------------------------------------------------------------

# 35. Current Version Assessment

The project history is now:

``` text
V1
 ↓
initial conditional GAN
 ↓
V2
 ↓
stable conditional GAN + trainable text encoder
 ↓
V2 epoch 500 baseline
 ↓
V3
 ↓
caption-conditioning evaluation
 ↓
V4
 ↓
stronger wrong-caption discrimination
 ↓
V5
 ↓
explicit image/text alignment loss
 ↓
V6
 ↓
multi-stage additive text conditioning
 ↓
V7
 ↓
FiLM multi-stage text conditioning
```

The main observed limitation remains:

``` text
generic landscape generation
```

rather than strong rendering of specific semantic concepts such as:

``` text
city skyline
road through mountains
forest with trees
beach with waves
lake with mountains
```

------------------------------------------------------------------------

# 36. New V8 Direction --- Dataset and Caption Quality

After V7, the next experiment should **not automatically add another
generator-conditioning mechanism**.

The evidence now points toward the training data and caption supervision
as major bottlenecks.

Current constraints include approximately:

``` text
~1,000 images
60×60 resolution
automatically generated BLIP captions
```

This makes it difficult for the GAN to learn precise object/scene
semantics.

The next step should therefore be a **dataset-first experiment**.

------------------------------------------------------------------------

# 37. V8 Step 1 --- Analyze Captions

Before training V8, inspect:

``` text
data/captions.txt
```

Analyze:

-   duplicate captions
-   overly generic captions
-   contradictory descriptions
-   very short captions
-   repeated concepts
-   useful scene/object vocabulary
-   frequency of important concepts

The goal is to determine whether the caption supervision is strong
enough for conditional generation.

------------------------------------------------------------------------

# 38. V8 Step 2 --- Improve Caption Quality

Instead of relying blindly on the raw BLIP output, construct a cleaner
and more consistent caption dataset.

Example target descriptions:

``` text
a mountain landscape with trees
a road through a mountain valley
a forest with dense green trees
a beach with ocean waves
a city skyline under a cloudy sky
```

The objective is to make the caption-image relationship clearer and more
consistent.

------------------------------------------------------------------------

# 39. V8 Step 3 --- Re-evaluate Resolution

After improving the captions, evaluate whether the image resolution is
limiting semantic learning.

Current:

``` text
60×60
```

Potential future experiments:

``` text
80×80
96×96
128×128
```

However, resolution should not be increased blindly on the CPU system.

Higher resolution increases:

-   computation
-   memory usage
-   training time
-   model complexity

The dataset and caption quality should be improved first.

------------------------------------------------------------------------

# 40. Model Preservation Strategy

Keep previous versions as experimental baselines.

Recommended preserved checkpoints:

``` text
V2 epoch 500
V4 epoch 650
V5 approximately epoch 750
V6 best/final checkpoint
V7 epoch 810
V7 epoch 820
V7 epoch 830
V7 epoch 950
```

Do not delete old checkpoints until the newer experiment has been
validated.

------------------------------------------------------------------------

# 41. Current Development Decision

The next development phase is:

``` text
V7 result
   ↓
stop architectural changes temporarily
   ↓
inspect captions.txt
   ↓
measure caption quality
   ↓
clean/improve captions
   ↓
rebuild training dataset
   ↓
V8 training experiment
```

The primary V8 objective is:

> Improve the quality of the image-caption supervision before
> introducing another major GAN architecture change.

------------------------------------------------------------------------

# 42. Current Evaluation Principle

For every future version, evaluate three separate properties:

### A. Training stability

``` text
Generator loss
Discriminator loss
No obvious collapse
```

### B. Visual quality

``` text
sharpness
structure
color
coherence
diversity
```

### C. Semantic conditioning

``` text
caption
   ↓
specific generated scene/object
```

The third property is the most important for the project's actual goal.

A model should not be considered better simply because:

``` text
loss decreased
```

or:

``` text
image became sharper
```

It should demonstrate stronger:

``` text
caption → semantic image
```

alignment.

------------------------------------------------------------------------

# 43. Current Status

``` text
Current completed version: V7

Best V7 visual region:
approximately epochs 810–830

Current major bottleneck:
caption/data supervision + limited dataset/resolution

Next planned version:
V8

V8 focus:
dataset and caption quality
```

The project should now move from repeated architecture experiments
toward improving the **quality and consistency of the training
supervision**.
