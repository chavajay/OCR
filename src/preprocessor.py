"""Image preprocessing module for OCR pipeline.

Provides robust preprocessing for real-world images including:
- Adaptive pipeline selection based on image quality
- CLAHE contrast enhancement for uneven lighting
- Adaptive thresholding for varying backgrounds
- Morphological operations for noise removal
- Deskew via projection profile analysis

Key design principle: minimal preprocessing for clean images,
aggressive preprocessing for noisy/degraded images.
"""

import cv2
import numpy as np


class ImagePreprocessor:
    """Handles image enhancement and binarization for OCR.

    Pipeline: Grayscale → [CLAHE] → [Denoise] → Binarize → [Morphology] → [Deskew]

    The [] steps are optional and applied adaptively based on image quality.
    """

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Converts BGR/RGB image to grayscale."""
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")
        if len(image.shape) == 2:
            return image
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"Unsupported shape: {image.shape}")

    @staticmethod
    def estimate_noise_level(gray: np.ndarray) -> float:
        """Estimates noise level using Laplacian variance.

        High variance = sharp/clean image.
        Low variance = blurry/noisy image.

        Returns:
            Noise estimate (higher = noisier). Range approximately 0-500.
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    @staticmethod
    def estimate_contrast(gray: np.ndarray) -> float:
        """Estimates contrast level.

        Returns:
            Contrast ratio (higher = more contrast). Range 0-1.
        """
        p5 = np.percentile(gray, 5)
        p95 = np.percentile(gray, 95)
        if p95 - p5 == 0:
            return 0.0
        return (p95 - p5) / 255.0

    @staticmethod
    def enhance_contrast(gray: np.ndarray) -> np.ndarray:
        """Applies CLAHE for contrast enhancement."""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def denoise_image(
        image: np.ndarray,
        method: str = "nlm",
        kernel_size: int = 3,
    ) -> np.ndarray:
        """Applies noise reduction."""
        if method == "median":
            return cv2.medianBlur(image, kernel_size)
        if method == "gaussian":
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        if method == "nlm":
            return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)
        return image

    @staticmethod
    def binarize_image(image: np.ndarray, method: str = "otsu", fixed_threshold: int = 127) -> np.ndarray:
        """Binarizes a grayscale image.

        Args:
            image: Grayscale image.
            method: 'fixed', 'otsu', 'adaptive', or 'sauvola'.
            fixed_threshold: Threshold value to use when method='fixed'.

        Returns:
            Binary image with values 0 and 255.
        """
        if len(image.shape) != 2:
            raise ValueError(f"Expected 2D image, got {image.shape}.")

        if method == "fixed":
            _, binary = cv2.threshold(
                image, fixed_threshold, 255, cv2.THRESH_BINARY_INV
            )
            return binary

        if method == "otsu":
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            return binary

        if method == "adaptive":
            return cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                15,
                8,
            )

        if method == "sauvola":
            return cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                25, 10
            )

        raise ValueError(f"Unknown binarization method: {method}")

    @staticmethod
    def morphological_clean(binary: np.ndarray) -> np.ndarray:
        """Applies morphological operations to clean up binary image."""
        kernel_small = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        kernel_med = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_med, iterations=1)
        return cleaned

    @staticmethod
    def correct_skew(image: np.ndarray, method: str = "projection") -> np.ndarray:
        """Corrects skew using projection profile analysis."""
        if len(image.shape) != 2:
            raise ValueError(f"Expected 2D image, got {image.shape}.")

        best_angle = 0.0
        best_var = 0.0
        search_range = (-10, 10) if method == "projection" else (-45, 45)
        steps = 41 if method == "projection" else 181

        for angle in np.linspace(search_range[0], search_range[1], steps):
            h, w = image.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            proj = np.sum(rotated, axis=1).astype(np.float64)
            var = np.var(proj)
            if var > best_var:
                best_var = var
                best_angle = angle

        if abs(best_angle) > 0.1:
            h, w = image.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), best_angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            image = (image > 0).astype(np.uint8) * 255

        return image

    def preprocess_adaptive(
        self,
        image: np.ndarray,
        deskew: bool = True,
        binarization: str = "fixed",
        fixed_threshold: int = 127,
    ) -> np.ndarray:
        """Binarizes image using a fixed threshold with optional denoising.

        Uses a fixed threshold (127) by default for deterministic binarization.
        This ensures training and inference produce identical character shapes
        (critical for the 18/784 pixel mismatch caused by global Otsu threshold
        varying between single-char and full-line canvases).

        For noisy images (Laplacian variance below 100), applies NLM denoising
        first to improve threshold quality.

        Args:
            image: Input image (H, W) or (H, W, 3).
            deskew: Whether to apply skew correction.
            binarization: 'fixed', 'otsu', 'adaptive', or 'sauvola'.
            fixed_threshold: Threshold value when binarization='fixed'.

        Returns:
            Preprocessed binary image (H, W), dtype uint8 with white text on black bg.
        """
        gray = self.to_grayscale(image)
        noise = self.estimate_noise_level(gray)

        if noise < 100:
            gray = self.denoise_image(gray, method="nlm")

        binary = self.binarize_image(gray, method=binarization, fixed_threshold=fixed_threshold)

        if deskew:
            binary = self.correct_skew(binary)

        return binary

    def preprocess(
        self,
        image: np.ndarray,
        binarization: str = "adaptive",
        denoise_method: str = "nlm",
        denoise_kernel: int = 3,
        deskew: bool = True,
        deskew_method: str = "projection",
        enhance: bool = True,
        clean_morphology: bool = True,
    ) -> np.ndarray:
        """Runs the full preprocessing pipeline.

        Args:
            image: Input image (H, W) or (H, W, 3).
            binarization: 'otsu', 'adaptive', or 'sauvola'.
            denoise_method: 'median', 'gaussian', or 'nlm'.
            denoise_kernel: Filter kernel size.
            deskew: Whether to apply skew correction.
            deskew_method: 'projection' or 'radon'.
            enhance: Whether to apply CLAHE contrast enhancement.
            clean_morphology: Whether to apply morphological cleaning.

        Returns:
            Preprocessed binary image (H, W), dtype uint8.
        """
        gray = self.to_grayscale(image)

        if enhance:
            gray = self.enhance_contrast(gray)

        denoised = self.denoise_image(gray, method=denoise_method, kernel_size=denoise_kernel)

        binary = self.binarize_image(denoised, method=binarization)

        if clean_morphology:
            binary = self.morphological_clean(binary)

        if deskew:
            binary = self.correct_skew(binary, method=deskew_method)

        return binary
