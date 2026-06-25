#!/usr/bin/env python3
"""Quantitative OCR evaluation suite.

Generates test images with known ground truth, runs full pipeline,
and computes character-level and word-level accuracy metrics.
"""

import os
import sys
import cv2
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import ImagePreprocessor
from src.segmenter import ProjectionSegmenter
from src.classifier import OCRClassifier
from scripts.generate_printed_dataset import CHARS, OPENCV_FONTS, render_char, random_augment


def render_full_line(text: str, font: int, scale: float, thickness: int) -> tuple[np.ndarray, list[str]]:
    """Renders a full text line as a single image (like a real document).
    Returns (image, list_of_chars_including_spaces)."""
    canvas_h = 60
    char_w = 48
    spacing = 4

    chars_info = []
    total_w = 0
    for ch in text:
        if ch == ' ':
            total_w += char_w // 2 + spacing
            chars_info.append(' ')
        else:
            total_w += char_w + spacing
            chars_info.append(ch)

    canvas = np.ones((canvas_h, total_w + 20), dtype=np.uint8) * 255
    x = 10
    for ch in text:
        if ch == ' ':
            x += char_w // 2 + spacing
        else:
            ts = cv2.getTextSize(ch, font, scale, thickness)[0]
            tx = x + (char_w - ts[0]) // 2
            ty = (canvas_h + ts[1]) // 2
            cv2.putText(canvas, ch, (tx, ty), font, scale, 0, thickness, cv2.LINE_AA)
            x += char_w + spacing

    return canvas, chars_info


def char_accuracy(pred: str, gt: str) -> float:
    """Character-level accuracy between two strings (aligned)."""
    if not gt:
        return 1.0 if not pred else 0.0
    max_len = max(len(pred), len(gt))
    correct = sum(1 for i in range(max_len) if i < len(pred) and i < len(gt) and pred[i] == gt[i])
    return correct / max_len


def run_baseline_eval():
    """Runs comprehensive evaluation across multiple test conditions."""
    pre = ImagePreprocessor()
    seg = ProjectionSegmenter()

    mapping = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?-()/\"' ")
    classifier = OCRClassifier(model_path="models/ocr_printed.pth", num_classes=len(mapping))
    classifier.load_model()
    l_idx = mapping.index('l')
    I_idx = mapping.index('I')

    test_texts = [
        "Hello World",
        "Hello OCR World",
        "Testing pipeline",
        "Python Code 123",
        "Quick Brown Fox",
        "The cat sat on mat",
        "ABCD efgh ijkl",
        "abc def ghi jkl",
        "0123456789",
        "Hello, World!",
        "Test 123.",
        "Machine Learning",
        "deep neural networks",
        "A B C D E F G",
        "a b c d e f g",
    ]

    results = {}

    # Test 1: Different OpenCV fonts
    for font_name, font in [("SIMPLEX", 0), ("DUPLEX", 2), ("COMPLEX", 3), ("TRIPLEX", 4), ("PLAIN", 1)]:
        for scale, thickness in [(0.6, 1), (0.8, 2), (1.0, 2)]:
            key = f"font_{font_name}_s{scale}_t{thickness}"
            total_chars = 0
            correct_chars = 0
            total_time = 0

            for text in test_texts:
                img, gt_chars = render_full_line(text, font, scale, thickness)
                binary = pre.preprocess(img, binarization="otsu", denoise_method="median",
                                        denoise_kernel=3, deskew=False, enhance=False, clean_morphology=False)
                chars_by_line, _ = seg.segment(binary)
                if not chars_by_line:
                    continue

                t0 = time.time()
                for line_chars in chars_by_line:
                    real = [c for c in line_chars if c.max() >= 0.01]
                    if real:
                        arr = np.stack(real, axis=0)
                        probs = classifier.predict_proba(arr)
                        preds = []
                        for p in probs:
                            top2 = np.argsort(p)[::-1][:2]
                            if l_idx in top2 and I_idx in top2 and abs(p[l_idx] - p[I_idx]) < 0.15:
                                preds.append(mapping[l_idx])
                            else:
                                preds.append(mapping[int(np.argmax(p))])
                        pred_text = ''.join(preds)
                        for i, c in enumerate(pred_text):
                            if i < len(gt_chars) and gt_chars[i] != ' ':
                                total_chars += 1
                                if i < len(pred_text) and pred_text[i] == gt_chars[i]:
                                    correct_chars += 1
                total_time += time.time() - t0

            acc = correct_chars / max(total_chars, 1)
            results[key] = {"acc": acc, "chars": total_chars, "correct": correct_chars, "time": total_time}

    # Test 2: With preprocessing variations
    for binarization in ["otsu", "adaptive"]:
        for denoise in ["median", "nlm"]:
            key = f"prep_{binarization}_{denoise}"
            total_chars = 0
            correct_chars = 0

            for text in test_texts:
                img, gt_chars = render_full_line(text, 0, 0.8, 2)
                binary = pre.preprocess(img, binarization=binarization, denoise_method=denoise,
                                        denoise_kernel=3, deskew=True, enhance=True, clean_morphology=True)
                chars_by_line, _ = seg.segment(binary)
                if not chars_by_line:
                    continue

                for line_chars in chars_by_line:
                    real = [c for c in line_chars if c.max() >= 0.01]
                    if real:
                        arr = np.stack(real, axis=0)
                        probs = classifier.predict_proba(arr)
                        preds = []
                        for p in probs:
                            top2 = np.argsort(p)[::-1][:2]
                            if l_idx in top2 and I_idx in top2 and abs(p[l_idx] - p[I_idx]) < 0.15:
                                preds.append(mapping[l_idx])
                            else:
                                preds.append(mapping[int(np.argmax(p))])
                        pred_text = ''.join(preds)
                        for i, c in enumerate(pred_text):
                            if i < len(gt_chars) and gt_chars[i] != ' ':
                                total_chars += 1
                                if i < len(pred_text) and pred_text[i] == gt_chars[i]:
                                    correct_chars += 1

            acc = correct_chars / max(total_chars, 1)
            results[key] = {"acc": acc, "chars": total_chars, "correct": correct_chars}

    # Test 3: Adversarial conditions
    for condition, transform in [
        ("noisy", lambda img: cv2.GaussianBlur(img, (3, 3), 0)),
        ("rotated_3deg", lambda img: cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 3, 1.0), (img.shape[1], img.shape[0]), borderValue=255)),
        ("small_font", lambda img: cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_AREA)),
    ]:
        total_chars = 0
        correct_chars = 0
        for text in test_texts:
            img, gt_chars = render_full_line(text, 0, 0.8, 2)
            img = transform(img)
            if condition == "small_font":
                img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2), interpolation=cv2.INTER_CUBIC)
            binary = pre.preprocess(img, binarization="otsu", denoise_method="median",
                                    denoise_kernel=3, deskew=True, enhance=True, clean_morphology=True)
            chars_by_line, _ = seg.segment(binary)
            if not chars_by_line:
                continue
            for line_chars in chars_by_line:
                real = [c for c in line_chars if c.max() >= 0.01]
                if real:
                    arr = np.stack(real, axis=0)
                    preds = classifier.predict(arr)
                    pred_text = ''.join(mapping[p] for p in preds)
                    for i, c in enumerate(pred_text):
                        if i < len(gt_chars) and gt_chars[i] != ' ':
                            total_chars += 1
                            if i < len(pred_text) and pred_text[i] == gt_chars[i]:
                                correct_chars += 1
        results[f"adv_{condition}"] = {"acc": correct_chars / max(total_chars, 1), "chars": total_chars, "correct": correct_chars}

    # Print results
    print("=" * 80)
    print("OCR BASELINE EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Condition':<45} {'Accuracy':>10} {'Chars':>8} {'Correct':>8}")
    print("-" * 80)

    best_key = max(results, key=lambda k: results[k]["acc"])
    worst_key = min(results, key=lambda k: results[k]["acc"])

    for key in sorted(results.keys()):
        r = results[key]
        acc_str = f"{r['acc']*100:.1f}%"
        marker = " <-- BEST" if key == best_key else (" <-- WORST" if key == worst_key else "")
        print(f"  {key:<43} {acc_str:>10} {r['chars']:>8} {r['correct']:>8}{marker}")

    print("-" * 80)
    all_accs = [r["acc"] for r in results.values()]
    print(f"  {'AVERAGE':<43} {np.mean(all_accs)*100:.1f}%")
    print(f"  {'BEST':<43} {results[best_key]['acc']*100:.1f}% ({best_key})")
    print(f"  {'WORST':<43} {results[worst_key]['acc']*100:.1f}% ({worst_key})")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_baseline_eval()
