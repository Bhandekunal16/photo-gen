"""Generate additional captions per image using BLIP (image-to-text).

Why this exists
---------------
The training set has only ~1000 images, one caption each, with many captions
being short and generic ("a body of water", "a black and white photo"). On
that data the text encoder (LSTM trained from scratch) has almost no signal
to learn a useful word->vector mapping, so text conditioning effectively
degenerates to noise.

This script uses Salesforce/blip-image-captioning-base (a small pretrained
vision-language model, ~250 MB) to generate N additional captions per image
with sampling. We then concatenate (existing caption + N BLIP captions) into
a new captions file. The result:

    * Same images, but each image now appears in the training set N+1 times
      paired with N+1 different textual descriptions.
    * BLIP captions are typically more grounded/specific than the originals
      ("a sailboat in the ocean at sunset" vs "a body of water").
    * Effective training-set size grows from 1000 to 1000*(N+1) pairs.

Usage
-----
    .venv/bin/python scripts/augment_captions.py \
        --image-dir data/image60px \
        --in-captions data/captions_60px.txt \
        --out-captions data/captions_60px_v2.txt \
        --extra-per-image 2

Runs on CPU. Expect ~2-5 seconds per image for 2 extra captions, i.e.
roughly 30-90 minutes for the full 1000-image dataset.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Keep TF/HF chatter down before importing them.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


MODEL_ID = "Salesforce/blip-image-captioning-base"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", required=True, type=Path)
    p.add_argument("--in-captions", required=True, type=Path)
    p.add_argument("--out-captions", required=True, type=Path)
    p.add_argument(
        "--extra-per-image",
        type=int,
        default=2,
        help="How many BLIP-generated captions to add per image (default: 2).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Max caption length in tokens (default: 20, matches MAX_LEN).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N images (for smoke tests).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed so reruns are reproducible.",
    )
    return p.parse_args()


def load_existing(captions_path: Path) -> list[tuple[str, str]]:
    """Read 'image_name|caption' lines, preserve order, skip blank/malformed."""
    pairs: list[tuple[str, str]] = []
    with open(captions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            name, caption = line.split("|", 1)
            pairs.append((name.strip(), caption.strip()))
    return pairs


def build_pipeline() -> tuple[BlipProcessor, BlipForConditionalGeneration]:
    """Load BLIP-base on CPU. Cached locally after first run."""
    print(f"Loading {MODEL_ID} (CPU, ~250MB on first run)...", flush=True)
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


@torch.no_grad()
def caption_image(
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    image: Image.Image,
    n: int,
    max_new_tokens: int,
) -> list[str]:
    """Generate `n` diverse captions for one image using nucleus sampling.

    We use sampling (not beam search) because we *want* diversity across the
    n captions for the same image. Top-p=0.9 plus a small temperature gives
    captions that vary in phrasing/focus but stay grounded.
    """
    inputs = processor(image.convert("RGB"), return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
        repetition_penalty=1.1,
    )
    captions = processor.batch_decode(outputs, skip_special_tokens=True)
    # De-duplicate while preserving order in case sampling collapses.
    seen: set[str] = set()
    unique: list[str] = []
    for c in captions:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not args.image_dir.is_dir():
        print(f"Image dir not found: {args.image_dir}", file=sys.stderr)
        return 1

    pairs = load_existing(args.in_captions)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} images to caption.")

    processor, model = build_pipeline()

    # Stream output so a crash mid-run still leaves a usable partial file.
    args.out_captions.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    started = time.time()
    with open(args.out_captions, "w", encoding="utf-8") as out:
        for idx, (name, original_caption) in enumerate(pairs, 1):
            img_path = args.image_dir / name
            if not img_path.exists():
                print(f"  [skip] missing: {img_path}", file=sys.stderr)
                continue

            image = Image.open(img_path)
            new_caps = caption_image(
                processor, model, image, args.extra_per_image, args.max_new_tokens
            )

            # Always include the human-authored caption first; it's the "anchor"
            # and may carry context the vision model can't see (proper nouns).
            all_caps = [original_caption] + new_caps
            for cap in all_caps:
                out.write(f"{name}|{cap}\n")
                written += 1
            out.flush()

            # Progress every ~10 images, with an ETA so the user can decide
            # whether to wait or come back later.
            if idx % 10 == 0 or idx == len(pairs):
                elapsed = time.time() - started
                rate = idx / elapsed
                eta = (len(pairs) - idx) / rate if rate > 0 else 0
                sys.stdout.write(
                    f"\r  {idx}/{len(pairs)}  "
                    f"({rate:.2f} img/s, ETA {eta/60:.1f} min)   "
                )
                sys.stdout.flush()

    sys.stdout.write("\n")
    print(f"Wrote {written} caption rows -> {args.out_captions}")
    print(f"Done in {(time.time() - started)/60:.1f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
