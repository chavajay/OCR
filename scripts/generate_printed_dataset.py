"""Generate synthetic printed character dataset matching the real pipeline.

Renders characters through the SAME preprocessing pipeline used at inference:
    Render char → place on white document → preprocess → segment → normalize

This ensures training data distribution matches what the classifier actually sees.
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?-()/\"' "

OPENCV_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
]

FONT_SCALES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
THICKNESSES = [1, 2, 3, 4]


def render_char_on_document(
    char: str,
    font: int,
    font_scale: float,
    thickness: int,
    canvas_w: int = 60,
    canvas_h: int = 60,
) -> np.ndarray:
    """Renders a single character centered on a white document canvas.

    This matches how characters appear in real document images before segmentation.
    Black text on white background.
    """
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255
    ts = cv2.getTextSize(char, font, font_scale, thickness)[0]
    x = max(0, (canvas_w - ts[0]) // 2)
    y = max(ts[1], (canvas_h + ts[1]) // 2)
    cv2.putText(canvas, char, (x, y), font, font_scale, 0, thickness, cv2.LINE_AA)
    return canvas


def normalize_for_classifier(char_img: np.ndarray, size: int = 28) -> np.ndarray:
    """Normalizes a character image to match classifier input format.

    This replicates the segmenter's _normalize_character exactly.
    """
    if char_img.size == 0:
        return np.zeros((size, size), dtype=np.float32)

    h, w = char_img.shape
    max_side = max(h, w)
    pad = max(2, int(max_side * 0.2))
    padded = np.zeros((max_side + 2 * pad, max_side + 2 * pad), dtype=np.uint8)
    y_offset = (max_side + 2 * pad - h) // 2
    x_offset = (max_side + 2 * pad - w) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = char_img
    resized = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    if normalized.mean() > 0.5:
        normalized = 1.0 - normalized
    normalized = (normalized > 0.2).astype(np.float32)
    return normalized


def augment_document_char(
    char_img: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Augments a character on a document canvas BEFORE normalization.

    Simulates real-world degradations at the document level:
    - Rotation (slight skew)
    - Noise (camera/scanner)
    - Blur (out of focus)
    - Brightness variation
    """
    h, w = char_img.shape

    # Slight rotation ±5°
    if rng.random() < 0.4:
        angle = rng.uniform(-5, 5)
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        char_img = cv2.warpAffine(char_img, M, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)

    # Gaussian noise
    if rng.random() < 0.5:
        sigma = rng.uniform(3, 15)
        noise = rng.normal(0, sigma, char_img.shape).astype(np.float32)
        char_img = np.clip(char_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Blur
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        char_img = cv2.GaussianBlur(char_img, (k, k), 0)

    # Brightness/contrast
    if rng.random() < 0.3:
        alpha = rng.uniform(0.8, 1.2)
        beta = rng.uniform(-10, 10)
        char_img = np.clip(alpha * char_img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Erosion/dilation (ink bleed)
    if rng.random() < 0.2:
        kernel = np.ones((2, 2), np.uint8)
        if rng.random() < 0.5:
            char_img = cv2.erode(char_img, kernel, iterations=1)
        else:
            char_img = cv2.dilate(char_img, kernel, iterations=1)

    return char_img


def generate_dataset(
    samples_per_char: int = 3000,
    output_dir: str = "data/printed",
) -> tuple:
    """Generates dataset with characters rendered through the document pipeline."""
    rng = np.random.default_rng(42)
    images = []
    labels = []

    total = len(CHARS) * samples_per_char
    count = 0

    for class_idx, char in enumerate(CHARS):
        for _ in range(samples_per_char):
            font = rng.choice(OPENCV_FONTS)
            font_scale = rng.choice(FONT_SCALES)
            thickness = rng.choice(THICKNESSES)

            # Step 1: Render on white document (black text on white bg)
            doc_img = render_char_on_document(char, font, font_scale, thickness)

            # Step 2: Augment at document level
            doc_img = augment_document_char(doc_img, rng)

            # Step 3: Normalize to match classifier input
            normalized = normalize_for_classifier(doc_img)

            images.append(normalized)
            labels.append(class_idx)
            count += 1

            if count % 10000 == 0:
                print(f"  Generated {count}/{total} samples...")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    perm = rng.permutation(len(images))
    images = images[perm]
    labels = labels[perm]

    split = int(0.9 * len(images))
    train_imgs, val_imgs = images[:split], images[split:]
    train_lbls, val_lbls = labels[:split], labels[split:]

    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, "printed_dataset.npz"),
        train_images=train_imgs,
        train_labels=train_lbls,
        val_images=val_imgs,
        val_labels=val_lbls,
    )

    print(f"\nDataset saved to {output_dir}/printed_dataset.npz")
    print(f"Train: {train_imgs.shape}, Val: {val_imgs.shape}")
    print(f"Classes: {len(CHARS)}")

    return train_imgs, train_lbls, val_imgs, val_lbls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pipeline-matched printed dataset")
    parser.add_argument("--samples", type=int, default=3000, help="Samples per character")
    parser.add_argument("--output", type=str, default="data/printed", help="Output directory")
    args = parser.parse_args()

    generate_dataset(samples_per_char=args.samples, output_dir=args.output)
