import os
import time

import boto3
import numpy as np
import requests
import torch
from io import BytesIO
from PIL import Image


def _create_fallback_tensors():
    """Return a blank 64×64 image + mask as a safe fallback on errors."""
    img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    return img, mask


class MidnightLook_LoadImageByURL:
    """Downloads an image from a URL and returns it as an IMAGE + MASK tensor."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"multiline": False, "default": "https://example.com/image.jpg"}),
            },
            "optional": {
                "return_mask": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "MidnightLook/Image"

    def load_image(self, url, return_mask=False):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content)).convert("RGB")

            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)[None, ...]

            h, w = img.height, img.width
            mask_tensor = torch.zeros((1, h, w), dtype=torch.float32)
            if return_mask:
                mask_tensor[:] = 1.0

            print(f"✅ MidnightLook (LoadImageByURL): Loaded image from {url}")
            return (img_tensor, mask_tensor)

        except requests.exceptions.RequestException as e:
            print(f"ERROR: MidnightLook_LoadImageByURL - Download failed: {e}")
            return _create_fallback_tensors()

        except Exception as e:
            print(f"ERROR: MidnightLook_LoadImageByURL - Unexpected error: {e}")
            return _create_fallback_tensors()


class MidnightLook_UploadToR2:
    """
    Uploads a single image to Cloudflare R2 and returns the public URL.

    Path format: ``{prefix}/{user_email}-{billing_id}/ML_{file_index}_{timestamp}.jpg``
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "bucket": ("STRING", {"multiline": False}),
                "endpoint_url": ("STRING", {"multiline": False}),
                "access_key": ("STRING", {"multiline": False}),
                "secret_key": ("STRING", {"multiline": True}),
                "billing_id": ("STRING", {"multiline": False, "default": "unknown"}),
                "user_email": ("STRING", {"multiline": False, "default": "unknown"}),
                "file_index": ("INT", {"default": 1, "min": 1, "max": 999}),
                "public_base_url": ("STRING", {"multiline": False, "default": ""}),
                "prefix": ("STRING", {"multiline": False, "default": "outputs"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "upload_image"
    CATEGORY = "MidnightLook/Image"
    OUTPUT_NODE = True

    def upload_image(
        self,
        image,
        bucket,
        endpoint_url,
        access_key,
        secret_key,
        billing_id,
        user_email,
        file_index,
        public_base_url,
        prefix="outputs",
    ):
        try:
            # 1. Create S3 client for R2
            s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="auto",
            )

            # 2. Tensor (1, H, W, C) → PIL Image
            img_tensor = image[0]
            img_np = img_tensor.cpu().numpy()
            img_pil = Image.fromarray(
                np.clip(img_np * 255.0, 0, 255).astype(np.uint8), "RGB"
            )

            # 3. Encode to JPEG in memory
            buffer = BytesIO()
            img_pil.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)

            # 4. Build the file key
            timestamp = int(time.time() * 1000)
            folder_path = f"{prefix}/{user_email}-{billing_id}"
            file_name = f"ML_{file_index}_{timestamp}.jpg"
            file_key = f"{folder_path}/{file_name}"

            # 5. Upload
            s3_client.upload_fileobj(
                buffer,
                bucket,
                file_key,
                ExtraArgs={"ContentType": "image/jpeg"},
            )

            # 6. Build the public URL
            if not public_base_url.strip():
                raise Exception("public_base_url is required but was not provided.")

            base = public_base_url.rstrip("/")
            url = f"{base}/{file_key.lstrip('/')}"

            print(f"✅ MidnightLook (UploadToR2): Uploaded image {file_index} → {url}")
            return (url,)

        except Exception as e:
            print(f"ERROR: MidnightLook_UploadToR2 - Upload failed: {e}")
            return (f"ERROR: {e}",)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "MidnightLook_LoadImageByURL": MidnightLook_LoadImageByURL,
    "MidnightLook_UploadToR2": MidnightLook_UploadToR2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MidnightLook_LoadImageByURL": "Load Image By URL (ML)",
    "MidnightLook_UploadToR2": "Upload Image to R2 (ML)",
}
