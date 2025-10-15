import torch
import numpy as np

class MidnightLook_CropForInpaint:
    """
    Crops an image and mask based on the mask's bounding box, resizes them, 
    and outputs the crop data for later use.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "target_width": ("INT", {"default": 1024, "min": 64, "step": 8}),
                "target_height": ("INT", {"default": 1024, "min": 64, "step": 8}),
                "padding": ("INT", {"default": 64, "min": 0, "step": 8, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA",)
    RETURN_NAMES = ("CROPPED_IMAGE", "CROPPED_MASK", "CROP_DATA",)
    FUNCTION = "crop_and_resize"
    CATEGORY = "MidnightLook/Inpaint"

    def crop_and_resize(self, image, mask, target_width, target_height, padding):
        # --- 1. Tensor to NumPy ---
        # ทำงานกับภาพและมาสก์แรกใน batch
        image_np = image[0].cpu().numpy()
        mask_np = mask[0].cpu().numpy()
        original_height, original_width, _ = image_np.shape

        # --- 2. Find Bounding Box from Mask ---
        if np.max(mask_np) == 0: # กรณีมาสก์ว่าง
            print("⚠️ MidnightLook (Crop): Mask is empty. Cropping the full image.")
            bbox = (0, 0, original_width, original_height)
        else:
            y_coords, x_coords = np.where(mask_np > 0)
            x_min, y_min = np.min(x_coords), np.min(y_coords)
            x_max, y_max = np.max(x_coords), np.max(y_coords)
            bbox = (x_min, y_min, x_max, y_max)

        # --- 3. Add Padding ---
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(original_width, x2 + padding + 1)
        y2 = min(original_height, y2 + padding + 1)

        # --- 4. Crop Image and Mask ---
        cropped_image_np = image_np[y1:y2, x1:x2, :]
        cropped_mask_np = mask_np[y1:y2, x1:x2]

        # --- 5. NumPy to Tensor for Resizing ---
        cropped_image_tensor = torch.from_numpy(cropped_image_np).unsqueeze(0)
        cropped_mask_tensor = torch.from_numpy(cropped_mask_np).unsqueeze(0)

        # --- 6. Resize with torch.nn.functional.interpolate ---
        # Image: [B, H, W, C] -> [B, C, H, W]
        img_for_resize = cropped_image_tensor.permute(0, 3, 1, 2)
        resized_image = torch.nn.functional.interpolate(
            img_for_resize, size=(target_height, target_width), mode='bicubic', align_corners=False
        )
        resized_image = resized_image.permute(0, 2, 3, 1) # Back to [B, H, W, C]

        # Mask: [B, H, W] -> [B, 1, H, W]
        mask_for_resize = cropped_mask_tensor.unsqueeze(1)
        resized_mask = torch.nn.functional.interpolate(
            mask_for_resize, size=(target_height, target_width), mode='nearest'
        )
        resized_mask = resized_mask.squeeze(1) # Back to [B, H, W]
        
        # --- 7. Create CROP_DATA for the paste-back node ---
        crop_data = (x1, y1, x2, y2, original_width, original_height)
        
        print(f"✅ MidnightLook (Crop): Cropped area [{x1}, {y1}, {x2}, {y2}] and resized to {target_width}x{target_height}")

        return (resized_image, resized_mask, (crop_data,))

class MidnightLook_PasteAfterInpaint:
    """
    Pastes a resized inpainted image back to its original location on the original image,
    using the CROP_DATA from the cropping node.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "inpainted_image": ("IMAGE",),
                "crop_data": ("CROP_DATA",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste_back"
    CATEGORY = "MidnightLook/Inpaint"

    def paste_back(self, original_image, inpainted_image, crop_data):
        # --- 1. Unpack CROP_DATA and get tensors ---
        data = crop_data[0]
        x1, y1, x2, y2, _, _ = data
        
        # .clone() to avoid modifying the original tensor in-place
        pasted_image = original_image.clone()
        
        # --- 2. Resize inpainted image back to original crop size ---
        crop_height = y2 - y1
        crop_width = x2 - x1
        
        # [B, H, W, C] -> [B, C, H, W] for interpolate
        inpainted_chw = inpainted_image.permute(0, 3, 1, 2)
        resized_inpainted = torch.nn.functional.interpolate(
            inpainted_chw, size=(crop_height, crop_width), mode='bicubic', align_corners=False
        )
        resized_inpainted = resized_inpainted.permute(0, 2, 3, 1) # Back to [B, H, W, C]

        # --- 3. Paste it back ---
        pasted_image[0, y1:y2, x1:x2, :] = resized_inpainted[0]
        
        print(f"✅ MidnightLook (Paste): Pasted inpainted image back to [{x1}, {y1}, {x2}, {y2}]")

        return (pasted_image,)
