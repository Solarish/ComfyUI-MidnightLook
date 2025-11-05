import torch
import requests
from PIL import Image
from io import BytesIO
import numpy as np

class MidnightLook_LoadImageByURL:
    """
    A custom node to load an image from a given URL.
    """
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

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    CATEGORY = "MidnightLook/Image"

    def load_image(self, url, return_mask):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            img = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Convert PIL Image to PyTorch Tensor
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)[None,] # Add batch dimension

            mask_tensor = torch.zeros((1, img.height, img.width), dtype=torch.float32)
            if return_mask:
                # If return_mask is true, create a full white mask
                mask_tensor = torch.ones((1, img.height, img.width), dtype=torch.float32)

            print(f"✅ MidnightLook (LoadImageByURL): Successfully loaded image from {url}")
            return (img_tensor, mask_tensor)

        except requests.exceptions.RequestException as e:
            print(f"ERROR: MidnightLook_LoadImageByURL - Failed to download image from URL: {e}")
            # Return a black image and empty mask on error
            return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64)))
        except Exception as e:
            print(f"ERROR: MidnightLook_LoadImageByURL - An unexpected error occurred: {e}")