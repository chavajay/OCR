#!/usr/bin/env python3
"""OCR Pipeline - Final Exam IA.

Usage:
    python main.py image.jpg [--output-dir ./output] [--model models/ocr_printed_best.pth]

Pipeline:
    Input Image → ImagePreprocessor → ProjectionSegmenter → OCRClassifier → ResultExporter
"""

import argparse
import sys
import os

import cv2
import numpy as np

from src.preprocessor import ImagePreprocessor
from src.segmenter import ProjectionSegmenter
from src.classifier import OCRClassifier
from src.exporter import ResultExporter


# Extended character mapping: digits + uppercase + lowercase + punctuation
CHAR_MAPPING = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?-()/\"' ")


def get_char_mapping() -> list[str]:
    """Returns the full character mapping for OCR classification.

    Maps class indices 0-74 to characters:
        Indices  0-9:   digits '0'-'9'
        Indices 10-35:  uppercase 'A'-'Z'
        Indices 36-61:  lowercase 'a'-'z'
        Indices 62-74:  punctuation and space

    Returns:
        List of 75 strings.
    """
    return CHAR_MAPPING


def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    image = cv2.imread(path)
    if image is None:
        raise ValueError(
            f"Failed to decode image: {path}. Supported: PNG, JPG, BMP, TIFF."
        )
    return image


def correct_common_errors(text: str) -> str:
    """Corrects systematic OCR confusions using word-level context.

    Rules are conservative — only correct when context is unambiguous.
    """
    result = list(text)
    for i, ch in enumerate(result):
        prev = result[i - 1] if i > 0 else ' '
        next_ch = result[i + 1] if i < len(result) - 1 else ' '

        # I → l: uppercase I between lowercase
        if ch == 'I' and prev.islower() and next_ch.islower():
            result[i] = 'l'

        # l → I: lowercase l between uppercase only at internal positions
        if ch == 'l' and prev.isupper() and next_ch.isupper():
            if prev not in (' ', '\t', '\n'):
                result[i] = 'I'

        # 0 → O: digit 0 between uppercase letters
        if ch == '0' and prev.isupper() and next_ch.isupper():
            result[i] = 'O'

        # O → 0: uppercase O between digits
        if ch == 'O' and prev.isdigit() and next_ch.isdigit():
            result[i] = '0'

        # c → C: lowercase c at word start
        if ch == 'c' and (prev == ' ' or prev == '\t' or prev == '\n'):
            result[i] = 'C'

        # w → W: lowercase w at word start
        if ch == 'w' and (prev == ' ' or prev == '\t' or prev == '\n'):
            result[i] = 'W'

        # o → O: lowercase o between uppercase letters (e.g., "OcR" → "OCR")
        if ch == 'o':
            if prev.isupper() and next_ch.isupper():
                result[i] = 'O'

    return ''.join(result)


def run_ocr(
    image_path: str,
    model_path: str = "models/ocr_printed_best.pth",
    output_dir: str = "output",
    binarization: str = "adaptive",
    denoise_method: str = "nlm",
) -> dict:
    """Runs the full OCR pipeline on an input image.

    Pipeline:
        1. Load image
        2. Preprocess (grayscale -> CLAHE -> denoise -> binarize -> morphology -> deskew)
        3. Segment (lines -> characters, each normalized to 28x28)
        4. Classify each character via CNN
        5. Map class indices to characters
        6. Post-process with context rules
        7. Export to .txt and .md

    Args:
        image_path: Path to the input image.
        model_path: Path to the trained model weights.
        output_dir: Directory for output files.
        binarization: Binarization method ('otsu', 'adaptive', 'sauvola').
        denoise_method: Denoising method ('median', 'gaussian', 'nlm').
    """
    # 1. Load
    image = load_image(image_path)
    print(f"Loaded image: {image.shape}")

    # 2. Preprocess - use fixed threshold (matches training pipeline)
    pre = ImagePreprocessor()
    binary = pre.preprocess_adaptive(image, deskew=True, binarization="fixed")
    print(f"Preprocessed: {binary.shape}")

    # 3. Segment
    seg = ProjectionSegmenter()
    chars_by_line, lines = seg.segment(binary)
    print(f"Detected {len(lines)} lines")

    n_chars = sum(len(chars) for chars in chars_by_line)
    print(f"Detected {n_chars} characters")

    if n_chars == 0:
        return {"text_lines": [], "txt_path": "", "md_path": ""}

    # 4. Classify
    mapping = get_char_mapping()
    classifier = OCRClassifier(model_path=model_path, num_classes=len(mapping))
    classifier.load_model()

    l_idx = mapping.index('l')
    I_idx = mapping.index('I')

    recognized_lines = []
    for chars in chars_by_line:
        if not chars:
            recognized_lines.append("")
            continue

        # Separate spaces from actual characters
        text_parts = []
        for c in chars:
            if c.max() < 0.01:
                text_parts.append(" ")
            else:
                text_parts.append(None)

        # Classify actual characters with probability-based l/I disambiguation
        real_chars = [c for c in chars if c.max() >= 0.01]
        if real_chars:
            char_array = np.stack(real_chars, axis=0)
            probs = classifier.predict_proba(char_array)
            pred_indices = []
            for p in probs:
                li_conf = p[l_idx]
                Ii_conf = p[I_idx]
                # If l and I are both top-2 and confidence gap < 15%, use context heuristic
                top2 = np.argsort(p)[::-1][:2]
                if l_idx in top2 and I_idx in top2 and abs(li_conf - Ii_conf) < 0.15:
                    # If adjacent chars suggest lowercase context, pick l
                    idx_in_top2 = list(top2).index(l_idx) if l_idx in top2 else -1
                    pred_indices.append(top2[0] if top2[0] == l_idx else l_idx)
                else:
                    pred_indices.append(int(np.argmax(p)))
            real_texts = [mapping[i] for i in pred_indices]
        else:
            real_texts = []

        # Reassemble text with spaces
        real_idx = 0
        line_text = ""
        for c in chars:
            if c.max() < 0.01:
                line_text += " "
            else:
                line_text += real_texts[real_idx]
                real_idx += 1

        line_text = correct_common_errors(line_text)
        recognized_lines.append(line_text)
        print(f"  Line {len(recognized_lines)}: {line_text}")

    # 5. Export
    exporter = ResultExporter(output_dir=output_dir)
    paths = exporter.export(recognized_lines)

    return {
        "text_lines": recognized_lines,
        "txt_path": paths["txt"],
        "md_path": paths["md"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR Pipeline - from image to text"
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument(
        "--model", type=str, default="models/ocr_printed_best.pth", help="Model weights path"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
    args = parser.parse_args()

    result = run_ocr(
        image_path=args.image,
        model_path=args.model,
        output_dir=args.output_dir,
    )

    print("\n=== OCR Results ===")
    for i, line in enumerate(result["text_lines"]):
        print(f"Line {i+1}: {line}")
    print(f"\nText exported to: {result['txt_path']}")
    print(f"Markdown exported to: {result['md_path']}")


if __name__ == "__main__":
    main()
