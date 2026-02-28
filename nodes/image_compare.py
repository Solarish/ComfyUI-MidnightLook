import os
import torch
import numpy as np
import folder_paths
from PIL import Image
import random
import string
import time

class MidnightLook_ImageCompare:
    """Takes two images, calculates similarity, and sends them to the UI widget."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image1", "image2")
    FUNCTION = "compare_images"
    CATEGORY = "MidnightLook/Image"
    OUTPUT_NODE = True

    def calculate_psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100.0
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

    def calculate_mae_percentage(self, img1, img2):
        mae = torch.mean(torch.abs(img1 - img2))
        percentage = (1.0 - mae.item()) * 100.0
        return percentage

    def compare_images(self, image1, image2):
        # Handle batch: we'll only do the visual compare on the first image in the batch
        # to prevent overloading the UI.
        
        # Ensure they are on CPU
        img1 = image1.cpu()
        img2 = image2.cpu()

        # Get first image of batch
        first_img1 = img1[0]
        first_img2 = img2[0]

        # Ensure sizes match for calculation. If not, resize image2 to image1's shape
        if first_img1.shape != first_img2.shape:
             first_img2 = torch.nn.functional.interpolate(
                first_img2.unsqueeze(0).permute(0, 3, 1, 2), 
                size=(first_img1.shape[0], first_img1.shape[1]), 
                mode="bilinear"
            ).permute(0, 2, 3, 1).squeeze(0)

        # Calculate metrics
        similarity_pct = self.calculate_mae_percentage(first_img1, first_img2)
        psnr_val = self.calculate_psnr(first_img1, first_img2)
        
        score_text = f"Similarity: {similarity_pct:.2f}% | PSNR: {psnr_val:.2f} dB"
        
        # Save temp files for the UI
        temp_dir = folder_paths.get_temp_directory()
        rand_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        timestamp = int(time.time() * 1000)
        
        img1_filename = f"ml_compare_{timestamp}_{rand_str}_1.png"
        img2_filename = f"ml_compare_{timestamp}_{rand_str}_2.png"
        
        img1_path = os.path.join(temp_dir, img1_filename)
        img2_path = os.path.join(temp_dir, img2_filename)
        
        # Convert to PIL to save easily
        i1_np = np.clip(first_img1.numpy() * 255.0, 0, 255).astype(np.uint8)
        i2_np = np.clip(first_img2.numpy() * 255.0, 0, 255).astype(np.uint8)
        
        Image.fromarray(i1_np).save(img1_path)
        Image.fromarray(i2_np).save(img2_path)
        
        # Format the UI output dict
        ui_output = {
            "bimgs": [
                {"filename": img1_filename, "type": "temp", "subfolder": ""},
                {"filename": img2_filename, "type": "temp", "subfolder": ""}
            ],
            "score": [score_text]
        }
        
        print(f"✅ MidnightLook (ImageCompare) -> {score_text}")

        return {"ui": ui_output, "result": (image1, image2)}
