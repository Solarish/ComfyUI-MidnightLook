"""
DeepFace Node for ComfyUI
==========================
Wrapper around the ``deepface`` library to provide robust face detection
using state-of-the-art backends (RetinaFace, YOLOv8, etc.).

Dependencies: deepface, numpy, torch, Pillow
"""

import numpy as np
import torch
from PIL import Image

# Lazy import to prevent startup crashes if deepface is missing
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False


class DeepFaceBBoxDetector:
    """
    Wrapper for DeepFace to act as a ComfyUI-Impact-Pack BBOX_DETECTOR.
    """
    def __init__(self, detector_backend: str, align: bool = False):
        self.detector_backend = detector_backend
        self.align = align

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        if not DEEPFACE_AVAILABLE:
            print("⚠️ DeepFace library not found.")
            return ((image.shape[1], image.shape[2]), [])

        # Impact Pack core / utils might be missing if user doesn't have it installed
        try:
            import impact.core as core
            from impact.core import SEG
            import impact.utils as utils
        except ImportError:
            print("⚠️ ComfyUI-Impact-Pack not found. BBOX_DETECTOR output is not fully functional.")
            return ((image.shape[1], image.shape[2]), [])

        drop_size = max(drop_size, 1)

        # image is (B, H, W, C). For Impact Pack, usually detect is called per-image (1, H, W, C)
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h, w, _ = img_np.shape

        try:
            detections = DeepFace.extract_faces(
                img_path=img_np,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=self.align,
                grayscale=False,
            )
        except Exception as e:
            print(f"⚠️ DeepFace error: {e}")
            return ((h, w), [])

        result = []
        for i, d in enumerate(detections):
            score = d.get("confidence", 1.0)
            if score > threshold:
                area = d["facial_area"]
                bx, by, bw, bh = area["x"], area["y"], area["w"], area["h"]
                
                # BBox format for Impact Pack is usually [x1, y1, x2, y2]
                x1, y1 = bx, by
                x2, y2 = bx + bw, by + bh

                if x2 - x1 > drop_size and y2 - y1 > drop_size:
                    item_bbox = [x1, y1, x2, y2]
                    
                    # Impact Pack utils.make_crop_region
                    crop_region = utils.make_crop_region(w, h, item_bbox, crop_factor)
                    
                    if detailer_hook is not None and hasattr(detailer_hook, "post_crop_region"):
                        crop_region = detailer_hook.post_crop_region(w, h, item_bbox, crop_region)
                        
                    crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
                    
                    # prepare cropped mask
                    cropped_mask = np.zeros((crop_y2 - crop_y1, crop_x2 - crop_x1))
                    
                    # Safe clamping for mask assignment
                    mask_y1 = max(0, y1 - crop_y1)
                    mask_y2 = min(cropped_mask.shape[0], y2 - crop_y1)
                    mask_x1 = max(0, x1 - crop_x1)
                    mask_x2 = min(cropped_mask.shape[1], x2 - crop_x1)
                    
                    if mask_y2 > mask_y1 and mask_x2 > mask_x1:
                        cropped_mask[mask_y1:mask_y2, mask_x1:mask_x2] = 1
                        
                    cropped_mask = utils.dilate_mask(cropped_mask, dilation)
                    
                    # label is string
                    label = "face"
                    item = SEG(None, cropped_mask, score, crop_region, item_bbox, label, None)
                    result.append(item)

        shape = h, w
        segs = shape, result

        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)

        return segs

    def detect_combined(self, image, threshold, dilation):
        try:
            from impact.core import segs_to_combined_mask
            return segs_to_combined_mask(self.detect(image, threshold, dilation, 1.0))
        except ImportError:
            import torch
            return torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32)

    def setAux(self, x):
        pass



class DeepFace_FaceCrop:
    """
    Detect and crop faces using the DeepFace library.
    Supports multiple backends: retinaface, yolov8, mediapipe, ssd, mtcnn.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detector_backend": (
                    [
                        "ssd",
                        "mtcnn",
                        "dlib",
                        "centerface",
                        "mediapipe",
                    ],
                    {"default": "ssd"},
                ),
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
                "force_square": ("BOOLEAN", {"default": True}),
                "align": ("BOOLEAN", {"default": False, "label": "Align Face"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "BBOX_DETECTOR")
    RETURN_NAMES = ("cropped_image", "mask", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "bbox_detector")
    FUNCTION = "crop_face"
    CATEGORY = "Midnight Look/Face"

    def crop_face(
        self,
        image: torch.Tensor,
        detector_backend: str,
        padding_factor: float,
        face_index: int,
        confidence_thresh: float,
        force_square: bool,
        align: bool,
    ):
        if not DEEPFACE_AVAILABLE:
            raise ImportError(
                "DeepFace library not found. Please install it with: "
                "pip install deepface"
            )

        # 1. Tensor -> Numpy (B, H, W, C) -> (H, W, C) RGB uint8
        batch_size = image.shape[0]
        if batch_size > 1:
            print(f"⚠️ DeepFace_FaceCrop: Processing first image only (Batch={batch_size})")

        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h_img, w_img, _ = img_np.shape

        # 2. Run DeepFace detection
        # extract_faces returns a list of dicts
        # target_size is ignored if enforce_detection=False usually, but 
        # let's just use the raw extraction.
        # DeepFace expects BGR if using opencv path, but numpy array is usually 
        # assumed RGB by some backends or BGR by others. 
        # DeepFace internals usually convert to BGR.
        
        try:
            detections = DeepFace.extract_faces(
                img_path=img_np,
                detector_backend=detector_backend,
                enforce_detection=False,  # Don't crash if no face
                align=align,
                grayscale=False,
            )
        except Exception as e:
            print(f"⚠️ DeepFace error: {e}")
            return self._passthrough(image, h_img, w_img)

        # Filter by confidence
        valid_detections = []
        for d in detections:
            # DeepFace returns 'confidence' key (0-1 approx, backend dependent)
            # Some backends might not return confidence, defaulting to 1.0
            score = d.get("confidence", 0.0)
            if score >= confidence_thresh:
                valid_detections.append(d)

        if not valid_detections:
            print(f"⚠️ DeepFace ({detector_backend}): No faces above threshold {confidence_thresh}")
            return self._passthrough(image, h_img, w_img)

        # 3. Sort by Area (largest first)
        # DeepFace returns 'facial_area': {'x': int, 'y': int, 'w': int, 'h': int}
        valid_detections.sort(
            key=lambda x: x["facial_area"]["w"] * x["facial_area"]["h"],
            reverse=True,
        )

        if face_index >= len(valid_detections):
            print(f"⚠️ Face index {face_index} out of range. Using 0.")
            face_index = 0

        target = valid_detections[face_index]
        area = target["facial_area"]
        
        # 4. Compute bbox with padding
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]
        
        # Convert to float for calculation
        abs_x, abs_y, abs_w, abs_h = float(x), float(y), float(w), float(h)
        
        pad_w = abs_w * padding_factor
        pad_h = abs_h * padding_factor

        x1 = abs_x - pad_w / 2
        y1 = abs_y - pad_h / 2
        x2 = abs_x + abs_w + pad_w / 2
        y2 = abs_y + abs_h + pad_h / 2

        # 5. Square crop
        if force_square:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            side = max(x2 - x1, y2 - y1)
            x1 = cx - side / 2
            y1 = cy - side / 2
            x2 = cx + side / 2
            y2 = cy + side / 2

        # 6. Clamp
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(w_img, x2))
        y2 = int(min(h_img, y2))
        
        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w <= 0 or crop_h <= 0:
            return self._passthrough(image, h_img, w_img)

        # 7. Crop
        cropped_np = img_np[y1:y2, x1:x2, :]

        # 8. Output
        cropped_tensor = (
            torch.from_numpy(cropped_np.astype(np.float32) / 255.0)
            .unsqueeze(0)
        )
        mask = torch.ones((1, crop_h, crop_w), dtype=torch.float32)

        print(
            f"✅ DeepFace ({detector_backend}): Cropped face at "
            f"[x={x1}, y={y1}, w={crop_w}, h={crop_h}]"
        )

        detector_obj = DeepFaceBBoxDetector(detector_backend=detector_backend, align=align)

        return (cropped_tensor, mask, x1, y1, crop_w, crop_h, detector_obj)

    @staticmethod
    def _passthrough(image, h, w):
        mask = torch.ones((1, h, w), dtype=torch.float32)
        detector_obj = DeepFaceBBoxDetector(detector_backend="none")
        return (image, mask, 0, 0, w, h, detector_obj)



class DeepFace_Verify:
    """
    Verify face identity using DeepFace models (commercial friendly).
    Models: Facenet, Facenet512, OpenFace, SFace.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "model_name": (
                    ["Facenet", "Facenet512", "OpenFace", "SFace"],
                    {"default": "Facenet512"},
                ),
                "detector_backend": (
                    ["ssd", "mtcnn", "dlib", "mediapipe", "retinaface"],
                    {"default": "ssd"},
                ),
                "distance_metric": (
                    ["cosine", "euclidean", "euclidean_l2"],
                    {"default": "cosine"},
                ),
                "threshold": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": (
                        "Cosine distance threshold. LOWER = STRICTER. "
                        "Facenet512 recommended: 0.30. "
                        "Values below threshold = Match."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("FLOAT", "BOOLEAN", "IMAGE", "STRING")
    RETURN_NAMES = ("distance", "is_match", "verified_image", "info_text")
    FUNCTION = "verify"
    CATEGORY = "Midnight Look/Face"
    OUTPUT_NODE = True

    # DeepFace recommended thresholds per model+metric
    # Source: deepface/commons/distance.py
    RECOMMENDED_THRESHOLDS = {
        ("Facenet", "cosine"): 0.40,
        ("Facenet", "euclidean"): 10.0,
        ("Facenet", "euclidean_l2"): 0.80,
        ("Facenet512", "cosine"): 0.30,
        ("Facenet512", "euclidean"): 23.56,
        ("Facenet512", "euclidean_l2"): 0.68,
        ("OpenFace", "cosine"): 0.10,
        ("OpenFace", "euclidean"): 0.55,
        ("OpenFace", "euclidean_l2"): 0.55,
        ("SFace", "cosine"): 0.593,
        ("SFace", "euclidean"): 10.734,
        ("SFace", "euclidean_l2"): 1.055,
    }

    def verify(self, image1, image2, model_name, detector_backend, distance_metric, threshold):
        if not DEEPFACE_AVAILABLE:
            raise ImportError("DeepFace library not found. pip install deepface")

        # Preprocess: Tensor [B,H,W,C] -> Numpy [H,W,C] (uint8)
        img1_np = (image1[0].cpu().numpy() * 255).astype(np.uint8)
        img2_np = (image2[0].cpu().numpy() * 255).astype(np.uint8)

        # Clean model_name (strip path prefixes)
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        if "\\" in model_name:
            model_name = model_name.split("\\")[-1]

        # Get recommended threshold for reference
        rec_thresh = self.RECOMMENDED_THRESHOLDS.get(
            (model_name, distance_metric), threshold
        )

        # Run Verify
        try:
            result = DeepFace.verify(
                img1_path=img1_np,
                img2_path=img2_np,
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                enforce_detection=False,
                align=True,
            )
        except Exception as e:
            print(f"⚠️ DeepFace Verify Error: {e}")
            return (1.0, False, torch.zeros_like(image2), f"Error: {e}")

        dist = result.get("distance", 1.0)
        is_match = dist <= threshold

        # Human-readable advice based on distance relative to recommended threshold
        if dist <= rec_thresh * 0.5:
            advice = "Identical"
        elif dist <= rec_thresh * 0.8:
            advice = "Very Similar"
        elif dist <= rec_thresh:
            advice = "Similar"
        elif dist <= rec_thresh * 1.5:
            advice = "Different"
        else:
            advice = "Very Different"

        icon = "✅" if is_match else "⛔"
        info = (
            f"{icon} Dist: {dist:.4f} ({advice}) | "
            f"Thresh: {threshold} | Recommended: {rec_thresh}"
        )
        print(f"[DeepFace] {info}")

        # Verified Image: Pass image2 if match, else black
        if is_match:
            verified_img = image2
        else:
            verified_img = torch.zeros_like(image2)

        return {"ui": {"text": [info]}, "result": (dist, is_match, verified_img, info)}


NODE_CLASS_MAPPINGS = {
    "DeepFace_FaceCrop": DeepFace_FaceCrop,
    "DeepFace_Verify": DeepFace_Verify,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepFace_FaceCrop": "Midnight Look : DeepFace Crop",
    "DeepFace_Verify": "Midnight Look : DeepFace Verify",
}

