"""Character segmentation module using MSER and connected components.

Implements robust text detection and character segmentation using:
- MSER (Maximally Stable Extremal Regions) for text detection
- Connected component analysis for character extraction
- Adaptive gap detection for word boundaries
"""

import cv2
import numpy as np


class ProjectionSegmenter:
    """Segments a binary document image into individual character matrices.

    Uses MSER for initial text region detection, then connected components
    for character extraction within detected regions.
    """

    def __init__(
        self,
        min_char_width: int = 2,
        min_char_height: int = 2,
        max_char_width: int = 150,
        max_char_height: int = 150,
        merge_x_threshold: int = 2,
        merge_y_threshold: int = 15,
    ):
        self.min_char_width = min_char_width
        self.min_char_height = min_char_height
        self.max_char_width = max_char_width
        self.max_char_height = max_char_height
        self.merge_x_threshold = merge_x_threshold
        self.merge_y_threshold = merge_y_threshold

    @staticmethod
    def detect_text_regions_mser(image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detects text regions using MSER (Maximally Stable Extremal Regions).

        MSER finds stable connected components across intensity thresholds,
        which is effective for detecting text in complex backgrounds.

        Args:
            image: Grayscale or binary image.

        Returns:
            List of (x, y, w, h) bounding boxes for detected text regions.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        mser = cv2.MSER_create(
            _min_area=30,
            _max_area=int(gray.shape[0] * gray.shape[1] * 0.1),
            _delta=5,
        )

        regions, _ = mser.detectRegions(gray)

        boxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            # Filter by size
            if w < 3 or h < 5:
                continue
            if w > gray.shape[1] * 0.5 or h > gray.shape[0] * 0.5:
                continue
            # Filter by aspect ratio (text characters are roughly 0.1-3.0)
            aspect = w / max(h, 1)
            if aspect > 5.0:
                continue
            boxes.append((x, y, w, h))

        return boxes

    def _connected_components(
        self, image: np.ndarray, max_area_ratio: float = 0.2
    ) -> list[tuple[int, int, int, int]]:
        """Extracts character bounding boxes via connected component analysis.

        Filters components by:
        - Minimum/maximum size
        - Aspect ratio (rejects very wide or very tall components)
        - Area relative to image size (rejects very large components like borders)

        Args:
            image: Binary image (H, W) with white text on black background.
            max_area_ratio: Maximum component area as fraction of image area.
                Use 0.2 for full-page images, 1.0 for cropped line images
                where characters naturally fill more of the frame.

        Returns:
            List of (x, y, w, h) bounding boxes for valid character components.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            image, connectivity=8
        )
        img_area = image.shape[0] * image.shape[1]
        boxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            # Basic size filter
            if w < self.min_char_width or h < self.min_char_height:
                continue
            if w > self.max_char_width or h > self.max_char_height:
                continue
            # Aspect ratio filter: reject very wide or very tall components
            aspect = w / max(h, 1)
            if aspect > 5.0 or aspect < 0.05:
                continue
            # Area filter: reject components too large relative to image
            if area > img_area * max_area_ratio:
                continue
            # Minimum area for text characters
            if area < 6:
                continue
            boxes.append((x, y, w, h))
        return boxes

    def _merge_close_boxes(
        self, boxes: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Merges close boxes that belong to the SAME character (e.g., 'i' dot + stem).

        Only merges when y-ranges do NOT overlap — this distinguishes:
        - Diacritic marks (dot '⏐', accent '´') from their base: y-ranges separated
        - Adjacent characters ('W' then 'o'): y-ranges OVERLAP → NOT merged
        """
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[0])
        merged = [boxes[0]]
        for x, y, w, h in boxes[1:]:
            px, py, pw, ph = merged[-1]
            # y-ranges overlap? (one starts before the other ends)
            y_overlap = py <= y + h and y <= py + ph
            horizontal_close = x <= px + pw + self.merge_x_threshold
            # Only merge if NOT overlapping in y (diacritic + base) AND close
            if not y_overlap and horizontal_close:
                new_x = min(px, x)
                new_y = min(py, y)
                new_w = max(px + pw, x + w) - new_x
                new_h = max(py + ph, y + h) - new_y
                merged[-1] = (new_x, new_y, new_w, new_h)
            else:
                merged.append((x, y, w, h))
        return merged

    def _detect_lines(
        self, boxes: list[tuple[int, int, int, int]]
    ) -> list[list[tuple[int, int, int, int]]]:
        """Groups bounding boxes into text lines based on y-range overlap.

        A component belongs to a line if its y-range overlaps with the line's
        bounding box. This correctly handles diacritic marks (dots, accents)
        that are vertically separated from the baseline but belong to the same
        text line as their base character.
        """
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[1])
        lines = [[boxes[0]]]
        for x, y, w, h in boxes[1:]:
            last_line = lines[-1]
            # Compute the y-range of the current line (min y to max y+h)
            line_y_min = min(b[1] for b in last_line)
            line_y_max = max(b[1] + b[3] for b in last_line)
            # Check if this component overlaps vertically with the line
            overlaps = y <= line_y_max and line_y_min <= y + h
            if overlaps:
                lines[-1].append((x, y, w, h))
            else:
                lines.append([(x, y, w, h)])
        return lines

    def segment_lines(self, image: np.ndarray) -> list[np.ndarray]:
        """Segments a binary document image into individual text lines.

        Uses connected component analysis to find character bounding
        boxes, groups them into lines, then merges within each line.
        """
        if image.size == 0:
            raise ValueError("Cannot segment an empty image.")
        if len(image.shape) != 2:
            raise ValueError(f"Expected 2D binary image, got shape {image.shape}.")

        boxes = self._connected_components(image)
        if not boxes:
            return []

        # Detect lines from raw boxes, then merge within each line
        lines = self._detect_lines(boxes)

        line_images = []
        for line_boxes in lines:
            line_boxes = self._merge_close_boxes(line_boxes)
            x_min = max(0, min(b[0] for b in line_boxes) - 2)
            y_min = max(0, min(b[1] for b in line_boxes) - 2)
            x_max = min(image.shape[1], max(b[0] + b[2] for b in line_boxes) + 2)
            y_max = min(image.shape[0], max(b[1] + b[3] for b in line_boxes) + 2)
            line_img = image[y_min:y_max, x_min:x_max]
            line_images.append(line_img)
        return line_images

    @staticmethod
    def _split_wide_component(
        char_img: np.ndarray, avg_width: float
    ) -> list[np.ndarray]:
        """Splits a wide component into individual characters via vertical projection.

        When characters touch (e.g. 'r' and 'n' merging into 'm'), connected
        components analysis sees them as a single blob. This method uses
        vertical projection to find natural split points — columns with
        minimal ink — and cuts the blob into separate characters.

        Args:
            char_img: Binary image of a single (potentially merged) component.
            avg_width: Average character width in the line, used as reference.

        Returns:
            List of character sub-images after splitting.
        """
        h, w = char_img.shape
        if w <= avg_width * 1.5:
            return [char_img]

        # Vertical projection: sum of ink per column
        proj = np.sum(char_img > 0, axis=0).astype(np.float32)

        # Smooth the projection to avoid noise
        kernel = np.ones(3) / 3.0
        proj_smooth = np.convolve(proj, kernel, mode='same')

        # Find columns with very little ink (valleys)
        threshold = proj_smooth.max() * 0.1
        valleys = proj_smooth < threshold

        # Find gaps (consecutive valley columns)
        splits = []
        i = 0
        while i < len(valleys):
            if valleys[i]:
                gap_start = i
                while i < len(valleys) and valleys[i]:
                    i += 1
                gap_end = i - 1
                gap_center = (gap_start + gap_end) // 2
                # Only split if gap is reasonably placed (not at edges)
                if gap_center > avg_width * 0.3 and gap_center < w - avg_width * 0.3:
                    splits.append(gap_center)
            else:
                i += 1

        if not splits:
            return [char_img]

        # Split at each valley
        result = []
        prev = 0
        for s in splits:
            part = char_img[:, prev:s]
            if part.shape[1] >= 3:
                result.append(part)
            prev = s
        last_part = char_img[:, prev:]
        if last_part.shape[1] >= 3:
            result.append(last_part)

        return result if result else [char_img]

    def segment_characters(self, line_image: np.ndarray) -> list[np.ndarray]:
        """Segments a text line image into individual character matrices.

        Detects word gaps (spaces) when horizontal distance between
        consecutive characters exceeds 1.2× the average character width.

        Uses relaxed area filter (max_area_ratio=1.0) because line images
        are already tightly cropped — characters naturally fill more of the frame.
        """
        if line_image.size == 0:
            raise ValueError("Cannot segment an empty line image.")

        boxes = self._connected_components(line_image, max_area_ratio=1.0)
        if not boxes:
            return []

        boxes = self._merge_close_boxes(boxes)
        boxes = sorted(boxes, key=lambda b: b[0])

        avg_width = np.mean([b[2] for b in boxes]) if boxes else 10

        characters = []
        for i, (x, y, w, h) in enumerate(boxes):
            if i > 0:
                prev_x, prev_y, prev_w, prev_h = boxes[i - 1]
                gap = x - (prev_x + prev_w)
                # Detect word-space gaps: in proportional fonts the space glyph
                # is ~0.3-0.5x avg_width; in monospace it's ~1.0x avg_width.
                # A threshold of 0.5x avg_width (min 6px) works for both.
                if gap > max(avg_width * 0.5, 6):
                    characters.append(np.zeros((28, 28), dtype=np.float32))

            char_img = line_image[y:y + h, x:x + w]
            splits = self._split_wide_component(char_img, avg_width)
            for part in splits:
                normalized = self._normalize_character(part)
                characters.append(normalized)
        return characters

    @staticmethod
    def _normalize_character(char_img: np.ndarray, size: int = 28) -> np.ndarray:
        """Normalizes and centers a character image into a fixed-size matrix.

        Pads to square, resizes to target size, normalizes to [0, 1].
        Returns float32 values preserving anti-aliasing gradients from the
        INTER_AREA resize. The classifier receives rich intensity information
        rather than binary pixels.

        Ensures white text on black background (matching training data format).
        Only inverts if mean > 0.7 (extremely unlikely for normal characters
        but guards against preprocessor polarity changes).
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

        # Invert only if extremely likely to be wrong polarity (mean > 0.7)
        # Normal characters never exceed 0.7 fill ratio in bounding box
        if normalized.mean() > 0.7:
            normalized = 1.0 - normalized

        return normalized

    def segment(self, image: np.ndarray) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
        """Complete segmentation pipeline: image → lines → characters."""
        lines = self.segment_lines(image)
        all_chars = []
        for line in lines:
            chars = self.segment_characters(line)
            all_chars.append(chars)
        return all_chars, lines
