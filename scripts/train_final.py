"""Final training: TrueType fonts + OpenCV fonts through the pipeline."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.preprocessor import ImagePreprocessor
from src.segmenter import ProjectionSegmenter
from src.classifier import OCRClassifier

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?-()/\"' "
N_CLASSES = len(CHARS)  # 75: digits + uppercase + lowercase + punctuation + space
SAMPLES_PER_CLASS = 800
EPOCHS = 80
BATCH_SIZE = 256

OPENCV_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
]
FONT_SCALES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
THICKNESSES = [1, 2, 3, 4]

TTF_FONTS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf',
    '/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf',
    '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
    '/usr/share/fonts/truetype/ubuntu/Ubuntu-RI.ttf',
    '/usr/share/fonts/truetype/ubuntu/Ubuntu-BI.ttf',
    '/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf',
    '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
    '/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf',
    '/usr/share/fonts/truetype/noto/NotoSans-Italic.ttf',
    '/usr/share/fonts/truetype/noto/NotoSans-BoldItalic.ttf',
    '/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf',
    '/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf',
    '/usr/share/fonts/truetype/noto/NotoSerif-Italic.ttf',
    '/usr/share/fonts/truetype/noto/NotoSerif-BoldItalic.ttf',
    '/usr/share/fonts/truetype/noto/NotoSansDisplay-Regular.ttf',
    '/usr/share/fonts/truetype/noto/NotoSerifDisplay-Regular.ttf',
    '/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
]
TTF_SIZES = [16, 20, 24, 28, 32, 36, 40]

pre = ImagePreprocessor()
seg = ProjectionSegmenter()


def render_opencv(char, font, scale, thick, rng):
    canvas = np.ones((60, 60), dtype=np.uint8) * 255
    ts = cv2.getTextSize(char, font, scale, thick)[0]
    x = rng.integers(2, max(3, 60 - ts[0]))
    y = rng.integers(ts[1] + 2, min(58, 60))
    cv2.putText(canvas, char, (x, y), font, scale, 0, thick, cv2.LINE_AA)
    return canvas


def render_pil(char, font_path, font_size, rng):
    """Render character at random position to simulate line extraction."""
    font = ImageFont.truetype(font_path, font_size)
    img = Image.new('L', (60, 60), 255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    max_x = max(2, 60 - tw - 2)
    max_y = max(th + 2, 58)
    x = rng.integers(0, max_x) - bbox[0]
    y = rng.integers(th + 2, max_y) - bbox[1]
    draw.text((x, y), char, fill=0, font=font)
    return np.array(img, dtype=np.uint8)


def add_realistic_degradations(canvas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Applies realistic image degradations to bridge synthetic-to-real gap.

    Simulates: camera noise, ink bleed, contrast variation, and JPEG artifacts.
    These degradations make synthetic training data more similar to real-world
    document images captured by cameras or scanners.
    """
    img = canvas.astype(np.float32)

    # Gaussian noise (camera sensor)
    if rng.random() < 0.5:
        sigma = rng.uniform(2, 8)
        noise = rng.normal(0, sigma, img.shape).astype(np.float32)
        img = img + noise

    # Contrast/brightness variation
    if rng.random() < 0.4:
        alpha = rng.uniform(0.7, 1.3)
        beta = rng.uniform(-15, 15)
        img = img * alpha + beta

    # Gaussian blur (out of focus / motion)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # JPEG compression artifacts (low quality save/load)
    if rng.random() < 0.3:
        quality = rng.integers(60, 95)
        ret, jpeg = cv2.imencode('.jpg', np.clip(img, 0, 255).astype(np.uint8),
                                [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ret:
            img = cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Erosion/dilation (ink bleed or faint print)
    if rng.random() < 0.2:
        kernel = np.ones((2, 2), np.uint8)
        binary = np.clip(img, 0, 255).astype(np.uint8)
        if rng.random() < 0.5:
            binary = cv2.erode(binary, kernel, iterations=1)
        else:
            binary = cv2.dilate(binary, kernel, iterations=1)
        img = binary.astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


def augment_threshold(char_img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly shifts the effective binarization threshold by ±2-3 levels.

    Simulates the 3-point Otsu threshold difference between isolated chars
    (threshold ~139) and line-extracted chars (threshold ~136). Randomly
    flips anti-aliased boundary pixels to make the model threshold-invariant.
    """
    if rng.random() < 0.5:
        return char_img
    offset = rng.uniform(-0.08, 0.08)
    img = char_img.astype(np.float32)
    img += offset
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def augment_stroke(char_img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly thin or thicken character strokes by 1 pixel.

    Preserves float values where possible — only binarizes temporarily for
    the morphological operation, then returns float32 result. This maintains
    gradient information that the CNN uses for fine discrimination.
    """
    if rng.random() < 0.7:
        binary = (char_img > 0.5).astype(np.uint8)
        kernel = np.ones((2, 2), np.uint8)
        if rng.random() < 0.5:
            binary = cv2.erode(binary, kernel, iterations=1)
        else:
            binary = cv2.dilate(binary, kernel, iterations=1)
        result = binary.astype(np.float32)
        # Blend with original float values to preserve gradient info
        if rng.random() < 0.3:
            result = (result + char_img * 1.0) * 0.5  # average
        return result
    return char_img


def process_through_pipeline(canvas):
    binary = pre.preprocess_adaptive(canvas, deskew=False, binarization="fixed")
    chars_by_line, _ = seg.segment(binary)
    if chars_by_line and chars_by_line[0]:
        for c in chars_by_line[0]:
            if c.max() >= 0.01:
                return c
    return None


def generate():
    rng = np.random.default_rng(42)
    images, labels = [], []
    total = N_CLASSES * SAMPLES_PER_CLASS
    count = 0
    t0 = time.time()

    half = SAMPLES_PER_CLASS // 2

    space_class_idx = CHARS.index(' ')

    for ci in range(N_CLASSES):
        # Special case: space character
        # In inference, spaces are injected as all-zero 28x28 arrays when gap
        # detection fires. Training must match this representation exactly.
        if ci == space_class_idx:
            zero_img = np.zeros((28, 28), dtype=np.float32)
            for _ in range(SAMPLES_PER_CLASS):
                zero_img_aug = zero_img.copy()
                zero_img_aug = augment_threshold(zero_img_aug, rng)
                images.append(zero_img_aug)
                labels.append(ci)
                count += 1
            continue

        char = CHARS[ci]

        # Half with OpenCV fonts
        for _ in range(half):
            font = rng.choice(OPENCV_FONTS)
            scale = rng.choice(FONT_SCALES)
            thick = rng.choice(THICKNESSES)
            canvas = render_opencv(char, font, scale, thick, rng)
            canvas = add_realistic_degradations(canvas, rng)
            img = process_through_pipeline(canvas)
            if img is None:
                continue
            img = augment_stroke(img, rng)
            img = augment_threshold(img, rng)
            images.append(img)
            labels.append(ci)
            count += 1

        # Half with TrueType fonts
        for _ in range(SAMPLES_PER_CLASS - half):
            font_path = rng.choice(TTF_FONTS)
            font_size = rng.choice(TTF_SIZES)
            canvas = render_pil(char, font_path, font_size, rng)
            canvas = add_realistic_degradations(canvas, rng)
            img = process_through_pipeline(canvas)
            if img is None:
                continue
            img = augment_stroke(img, rng)
            img = augment_threshold(img, rng)
            images.append(img)
            labels.append(ci)
            count += 1

        if (ci + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = count / elapsed if elapsed > 0 else 0
            print(f"  Class {ci+1}/{N_CLASSES}, {count} samples ({rate:.0f}/s)", flush=True)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    perm = rng.permutation(len(images))
    images, labels = images[perm], labels[perm]
    split = int(0.9 * len(images))
    print(f"\nGenerated {len(images)} total samples", flush=True)
    return images[:split], labels[:split], images[split:], labels[split:]


def main():
    print("=" * 60, flush=True)
    print("FINAL TRAINING: OpenCV + TrueType fonts through pipeline", flush=True)
    print(f"Classes: {N_CLASSES}, Target/class: {SAMPLES_PER_CLASS}", flush=True)
    print(f"TTF fonts: {len(TTF_FONTS)}, TTF sizes: {TTF_SIZES}", flush=True)
    print(f"OpenCV fonts: {len(OPENCV_FONTS)}", flush=True)
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    print("\nGenerating data...", flush=True)
    ti, tl, vi, vl = generate()
    gen_time = time.time() - t0
    print(f"Data generated in {gen_time:.0f}s", flush=True)
    print(f"Train: {ti.shape}, Val: {vi.shape}", flush=True)

    print("\nTraining CNN...", flush=True)
    classifier = OCRClassifier(model_path="models/ocr_printed.pth", num_classes=N_CLASSES)
    train_t0 = time.time()
    history = classifier.train_model(
        train_data=ti, train_labels=tl,
        val_data=vi, val_labels=vl,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        learning_rate=0.001, weight_decay=1e-4,
        augment=False,
    )
    train_time = time.time() - train_t0

    best_val = max(history["val_acc"])
    final_val = history["val_acc"][-1]

    print("\n" + "=" * 60, flush=True)
    print(f"COMPLETE", flush=True)
    print(f"Gen: {gen_time:.0f}s, Train: {train_time:.0f}s ({train_time/60:.1f}min)", flush=True)
    print(f"Best val acc: {best_val:.4f} ({best_val*100:.2f}%)", flush=True)
    print(f"Final val acc: {final_val:.4f} ({final_val*100:.2f}%)", flush=True)
    print("=" * 60, flush=True)
    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
