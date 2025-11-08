import torch
import requests
from PIL import Image
from io import BytesIO
import numpy as np

class MidnightLook_LoadImageByURL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"multiline": False, "default": "https://example.com/image.jpg"}),
            },
            "optional": {
                "return_mask": ("BOOLEAN", {"default": False}),
            }
        }

    # <<< ComfyUI expects IMAGE as [B, H, W, C] and MASK as [B, H, W]
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "MidnightLook/Image"

    def load_image(self, url, return_mask=False):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content)).convert("RGB")

            # H, W, C
            img_np = np.array(img).astype(np.float32) / 255.0
            # B, H, W, C
            img_tensor = torch.from_numpy(img_np)[None, ...]

            # MASK: [B, H, W]
            h, w = img.height, img.width
            mask_tensor = torch.zeros((1, h, w), dtype=torch.float32)
            if return_mask:
                mask_tensor[:] = 1.0

            print(f"✅ MidnightLook (LoadImageByURL): Successfully loaded image from {url}")
            return (img_tensor, mask_tensor)

        except requests.exceptions.RequestException as e:
            print(f"ERROR: MidnightLook_LoadImageByURL - Failed to download image from URL: {e}")
            # fallback 64x64 black
            img_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            mask_tensor = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (img_tensor, mask_tensor)

        except Exception as e:
            print(f"ERROR: MidnightLook_LoadImageByURL - An unexpected error occurred: {e}")
            img_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            mask_tensor = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (img_tensor, mask_tensor)
