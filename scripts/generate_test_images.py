"""Generate test images for OCR validation.

Creates multiple test images with different fonts, sizes, and text content
to thoroughly test the OCR pipeline on various printed text scenarios.

Characters are rendered individually with proper spacing (like training data)
to ensure the segmentation pipeline can separate them.

Usage:
    python scripts/generate_test_images.py
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test cases: (text_lines, font, scale, thickness, description)
TEST_CASES = [
    # Case 1: Simple lowercase
    (["hello world", "python code"], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, "simple_lower"),
    # Case 2: Uppercase
    (["HELLO WORLD", "PYTHON CODE"], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, "uppercase"),
    # Case 3: Mixed case
    (["Hello World", "Python Code"], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, "mixed_case"),
    # Case 4: Numbers
    (["1234567890", "Phone: 555-1234"], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, "numbers"),
    # Case 5: Mixed alpha-numeric
    (["Test123 Abc456", "Code789 Data000"], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, "alphanum"),
    # Case 6: Different font - DUPLEX
    (["Quick Brown Fox", "Jumps Over Dog"], cv2.FONT_HERSHEY_DUPLEX, 0.8, 2, "duplex"),
    # Case 7: Different font - COMPLEX
    (["Machine Learning", "Deep Networks"], cv2.FONT_HERSHEY_COMPLEX, 0.8, 2, "complex"),
    # Case 8: Small text
    (["small text sample", "tiny font test"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, "small_text"),
    # Case 9: Large text
    (["LARGE TEXT", "BIG FONT"], cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3, "large_text"),
    # Case 10: Multiple lines
    (["Line one here", "Line two here", "Line three here"], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2, "multi_line"),
    # Case 11: Programming code-like
    (["def function():", "return result"], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2, "code_like"),
    # Case 12: Sentence with punctuation
    (["Hello, World!", "Test 123."], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, "punctuation"),
    # Case 13: TRIPLEX font
    (["Science Technology", "Engineering Math"], cv2.FONT_HERSHEY_TRIPLEX, 0.7, 2, "triplex"),
    # Case 14: Thin font (PLAIN)
    (["Light weight text", "Thin strokes"], cv2.FONT_HERSHEY_PLAIN, 1.0, 1, "plain_thin"),
    # Case 15: Different thicknesses
    (["Bold Title Here", "normal subtitle"], cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3, "mixed_weight"),
]


def render_char_individual(
    char: str,
    font: int,
    font_scale: float,
    thickness: int,
    canvas_size: int = 48,
) -> np.ndarray:
    """Renders a single character centered on a white canvas (like a real document).

    Draws black text on white background to match real-world document format.
    The OCR pipeline's preprocessor uses BINARY_INV to convert this to
    white-text-on-black (matching training data format).

    Args:
        char: Single character to render.
        font: OpenCV font constant.
        font_scale: Font size multiplier.
        thickness: Stroke thickness.
        canvas_size: Size of rendering canvas (default 48).

    Returns:
        48x48 grayscale image with black text on white background.
    """
    canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    text_size = cv2.getTextSize(char, font, font_scale, thickness)[0]
    x = (canvas_size - text_size[0]) // 2
    y = (canvas_size + text_size[1]) // 2
    cv2.putText(canvas, char, (x, y), font, font_scale, 0, thickness, cv2.LINE_AA)
    return canvas


def render_text_lines(
    lines: list[str],
    font: int,
    font_scale: float,
    thickness: int,
    padding: int = 20,
    char_spacing: int = 8,
    line_spacing: int = 25,
) -> np.ndarray:
    """Renders multiple text lines with individually spaced characters.

    Each character is rendered on a 48x48 canvas (like training data) and
    placed with proper spacing to ensure the segmentation pipeline can
    separate them. The whitespace around each character matches the
    training data distribution.

    Args:
        lines: List of text strings to render.
        font: OpenCV font constant.
        font_scale: Font size multiplier.
        thickness: Stroke thickness.
        padding: Padding around text in pixels.
        char_spacing: Horizontal spacing between characters in pixels.
        line_spacing: Vertical spacing between lines in pixels.

    Returns:
        Binary image (H, W) with white background and black text.
    """
    canvas_size = 48  # Each character canvas size

    # First pass: calculate image dimensions
    char_images = []
    max_line_height = 0
    total_width = 0

    for line in lines:
        line_chars = []
        line_width = 0
        line_height = 0
        for ch in line:
            if ch == ' ':
                # Space: use half canvas width
                space_w = canvas_size // 2
                line_width += space_w + char_spacing
                line_chars.append(None)
            else:
                char_img = render_char_individual(ch, font, font_scale, thickness)
                line_chars.append(char_img)
                line_width += canvas_size + char_spacing
                line_height = max(line_height, canvas_size)
        char_images.append(line_chars)
        max_line_height = max(max_line_height, line_height)
        total_width = max(total_width, line_width)

    # Create canvas
    img_width = total_width + 2 * padding
    img_height = len(lines) * (max_line_height + line_spacing) + 2 * padding
    canvas = np.ones((img_height, img_width), dtype=np.uint8) * 255

    # Second pass: render characters
    for li, line_chars in enumerate(char_images):
        y_offset = padding + li * (max_line_height + line_spacing)
        x_offset = padding
        for char_img in line_chars:
            if char_img is None:
                # Space
                space_w = canvas_size // 2
                x_offset += space_w + char_spacing
            else:
                # Place 48x48 character canvas
                y_pos = y_offset + (max_line_height - canvas_size) // 2
                canvas[y_pos:y_pos + canvas_size, x_offset:x_offset + canvas_size] = char_img
                x_offset += canvas_size + char_spacing

    return canvas


def main():
    """Generates all test images and saves to tests/images/."""
    output_dir = "tests/images"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {len(TEST_CASES)} test images...")
    print(f"Output directory: {output_dir}\n")

    for lines, font, scale, thickness, name in TEST_CASES:
        img = render_text_lines(lines, font, scale, thickness)
        filepath = os.path.join(output_dir, f"test_{name}.png")
        cv2.imwrite(filepath, img)
        print(f"  [{name}] {img.shape[1]}x{img.shape[0]}px - {lines}")

    # Also create a challenging image with all characters
    challenging_lines = ["abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789"]
    img = render_text_lines(challenging_lines, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2, padding=15, char_spacing=6)
    filepath = os.path.join(output_dir, "test_all_chars.png")
    cv2.imwrite(filepath, img)
    print(f"  [all_chars] {img.shape[1]}x{img.shape[0]}px - full character set")

    # Create an image with rotated text (slight angle)
    base = render_text_lines(["Rotated Text Sample"], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    h, w = base.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 5, 1.0)
    rotated = cv2.warpAffine(base, M, (w, h), borderValue=255)
    filepath = os.path.join(output_dir, "test_rotated.png")
    cv2.imwrite(filepath, rotated)
    print(f"  [rotated] {w}x{h}px - 5 degree rotation")

    print(f"\nDone! {len(TEST_CASES) + 2} test images generated.")
    print(f"\nRun: python main.py tests/images/test_<name>.png")


if __name__ == "__main__":
    main()
