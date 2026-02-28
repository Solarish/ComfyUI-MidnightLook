"""
MediaPipe Face Crop Node for ComfyUI
=====================================
Uses the **MediaPipe Tasks** API (2024+) for face detection and cropping.
Legacy ``mp.solutions`` API is not used.

Supports both **Short Range** (blaze_face_short_range.tflite) and
**Full Range** (blaze_face_full_range.tflite) models.

Dependencies: mediapipe, torch, numpy
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np
import torch

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceDetector,
    FaceDetectorOptions,
    RunningMode,
)

# ---------------------------------------------------------------------- #
#  Model management
# ---------------------------------------------------------------------- #
_MODELS = {
    "short_range": {
        "filename": "blaze_face_short_range.tflite",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/latest/"
            "blaze_face_short_range.tflite"
        ),
    },
    # Note: Full range model URL follows the same pattern
    "full_range": {
        "filename": "blaze_face_full_range.tflite",
        "url": (
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_full_range/float16/latest/"
            "blaze_face_full_range.tflite"
        ),
    },
}


def _resolve_model_path(model_type: str = "short_range") -> str:
    """Return the absolute path to the TFLite model, downloading it if
    it does not exist yet."""
    model_info = _MODELS.get(model_type)
    if not model_info:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Store model in the same directory as this script
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, model_info["filename"])

    if not os.path.isfile(model_path):
        print(
            f"⬇️  MediaPipe_FaceCrop: Downloading {model_type} model "
            f"to {model_path} …"
        )
        try:
            urllib.request.urlretrieve(model_info["url"], model_path)
            print(f"✅  MediaPipe_FaceCrop: {model_type} model ready.")
        except Exception as e:
            print(f"❌  MediaPipe_FaceCrop: Download failed: {e}")
            raise e

    return model_path


# ---------------------------------------------------------------------- #
#  Node class
# ---------------------------------------------------------------------- #
class MediaPipe_FaceCrop:
    """Detect and crop faces using Google MediaPipe Face Detection
    (Tasks API). Now supports Full Range model for distant faces."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_type": (["short_range", "full_range"],),
                "padding_factor": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "face_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 99, "step": 1},
                ),
                "confidence_thresh": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "force_square": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "mask", "bbox_x", "bbox_y", "bbox_w", "bbox_h")
    FUNCTION = "crop_face"
    CATEGORY = "Midnight Look/Face"

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #
    def crop_face(
        self,
        image: torch.Tensor,
        model_type: str,
        padding_factor: float,
        face_index: int,
        confidence_thresh: float,
        force_square: bool,
    ):
        # -------------------------------------------------------------- #
        # 1. Tensor -> Numpy
        # -------------------------------------------------------------- #
        batch_size = image.shape[0]
        if batch_size > 1:
            print(
                f"⚠️  MediaPipe_FaceCrop: batch size is {batch_size}, "
                "processing only the first image."
            )

        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h, w, _ = img_np.shape

        # -------------------------------------------------------------- #
        # 2. Run face detection
        # -------------------------------------------------------------- #
        model_path = _resolve_model_path(model_type)

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=confidence_thresh,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_np,
        )

        with FaceDetector.create_from_options(options) as detector:
            result = detector.detect(mp_image)

        # -------------------------------------------------------------- #
        # 3. Handle no-detection
        # -------------------------------------------------------------- #
        if not result.detections:
            print(
                f"⚠️  MediaPipe_FaceCrop ({model_type}): No face detected."
            )
            return self._passthrough(image, h, w)

        # -------------------------------------------------------------- #
        # 4. Sort & Select
        # -------------------------------------------------------------- #
        detections = sorted(
            result.detections,
            key=lambda d: d.bounding_box.width * d.bounding_box.height,
            reverse=True,
        )

        if face_index >= len(detections):
            face_index = 0

        bbox = detections[face_index].bounding_box

        # Absolute coordinates
        abs_x = float(bbox.origin_x)
        abs_y = float(bbox.origin_y)
        abs_w = float(bbox.width)
        abs_h = float(bbox.height)

        # -------------------------------------------------------------- #
        # 5. Padding
        # -------------------------------------------------------------- #
        pad_w = abs_w * padding_factor
        pad_h = abs_h * padding_factor

        x1 = abs_x - pad_w / 2
        y1 = abs_y - pad_h / 2
        x2 = abs_x + abs_w + pad_w / 2
        y2 = abs_y + abs_h + pad_h / 2

        # -------------------------------------------------------------- #
        # 6. Force Square
        # -------------------------------------------------------------- #
        if force_square:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            side = max(x2 - x1, y2 - y1)
            x1 = cx - side / 2
            y1 = cy - side / 2
            x2 = cx + side / 2
            y2 = cy + side / 2

        # -------------------------------------------------------------- #
        # 7. Clamp & Crop
        # -------------------------------------------------------------- #
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(w, x2))
        y2 = int(min(h, y2))

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w <= 0 or crop_h <= 0:
            return self._passthrough(image, h, w)

        cropped_np = img_np[y1:y2, x1:x2, :]

        # -------------------------------------------------------------- #
        # 8. Output
        # -------------------------------------------------------------- #
        cropped_tensor = (
            torch.from_numpy(cropped_np.astype(np.float32) / 255.0)
            .unsqueeze(0)
        )
        mask = torch.ones((1, crop_h, crop_w), dtype=torch.float32)

        print(
            f"✅ MediaPipe_FaceCrop: Cropped face ({model_type}) at "
            f"[x={x1}, y={y1}, w={crop_w}, h={crop_h}]"
        )

        return (cropped_tensor, mask, x1, y1, crop_w, crop_h)

    @staticmethod
    def _passthrough(image: torch.Tensor, h: int, w: int):
        mask = torch.ones((1, h, w), dtype=torch.float32)
        return (image, mask, 0, 0, w, h)


NODE_CLASS_MAPPINGS = {
    "MediaPipe_FaceCrop": MediaPipe_FaceCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipe_FaceCrop": "Midnight Look : MediaPipe Face Crop",
}
