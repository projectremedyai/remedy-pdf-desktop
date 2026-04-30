"""Pillow-based image contrast enhancement for embedded PDF images."""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)


class ImageContrastEnhancer:
    """Enhances contrast of embedded images using Pillow."""

    def enhance(
        self,
        image_bytes: bytes,
        target_increase: float = 1.3,
        brightness_adjust: float = 1.0,
    ) -> bytes:
        """Increase contrast of an image.

        Args:
            image_bytes: Input image as bytes (any format Pillow supports).
            target_increase: Contrast enhancement factor (1.0 = no change,
                1.3 = 30% increase).
            brightness_adjust: Optional brightness adjustment (1.0 = no change).

        Returns:
            Enhanced image as PNG bytes.
        """
        from PIL import Image, ImageEnhance

        img = Image.open(io.BytesIO(image_bytes))

        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(target_increase)

        if brightness_adjust != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_adjust)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def analyze_contrast(self, image_bytes: bytes) -> dict:
        """Compute contrast metrics for an image.

        Returns dict with:
            rms_contrast: Root-mean-square contrast (0-1)
            dynamic_range: Ratio of (max - min) to 255
            mean_luminance: Average brightness (0-255)
        """
        from PIL import Image
        import statistics

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "L":
            img = img.convert("L")

        pixels = list(img.getdata())
        if not pixels:
            return {"rms_contrast": 0.0, "dynamic_range": 0.0, "mean_luminance": 0.0}

        mean = statistics.mean(pixels)
        rms = (statistics.mean((p - mean) ** 2 for p in pixels)) ** 0.5 / 255.0
        min_val = min(pixels)
        max_val = max(pixels)
        dynamic_range = (max_val - min_val) / 255.0

        return {
            "rms_contrast": round(rms, 4),
            "dynamic_range": round(dynamic_range, 4),
            "mean_luminance": round(mean, 2),
        }
