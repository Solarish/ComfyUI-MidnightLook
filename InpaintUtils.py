import torch
import numpy as np

class MidnightLook_CropForInpaint:
    """
    Crops a square region from the image based on the mask's bounding box,
    resizes it to a 1:1 aspect ratio, and outputs crop data.
    This version crops to the smaller dimension to avoid black bars (zoom crop).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "target_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "padding": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA",)
    RETURN_NAMES = ("CROPPED_IMAGE", "CROPPED_MASK", "CROP_DATA",)
    FUNCTION = "crop_and_resize"
    CATEGORY = "MidnightLook/Inpaint"

    def crop_and_resize(self, image, mask, target_size, padding):
        # --- 1. Tensor to NumPy ---
        image_np = image[0].cpu().numpy()
        mask_np = mask[0].cpu().numpy()
        original_height, original_width, _ = image_np.shape

        # --- 2. Find Bounding Box from Mask ---
        if np.max(mask_np) == 0:
            bbox = (0, 0, original_width, original_height)
        else:
            y_coords, x_coords = np.where(mask_np > 0)
            x_min, y_min = np.min(x_coords), np.min(y_coords)
            x_max, y_max = np.max(x_coords), np.max(y_coords)
            bbox = (x_min, y_min, x_max, y_max)

        # --- 3. Add Padding to BBox ---
        x1, y1, x2, y2 = bbox
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(original_width, x2 + padding)
        y2_padded = min(original_height, y2 + padding)
        
        crop_width = x2_padded - x1_padded
        crop_height = y2_padded - y1_padded

        # --- 4. New Logic: Crop a square from the center of the padded area ---
        side_length = min(crop_width, crop_height)
        center_x = x1_padded + crop_width / 2.0
        center_y = y1_padded + crop_height / 2.0

        square_x1 = int(center_x - side_length / 2.0)
        square_y1 = int(center_y - side_length / 2.0)
        square_x2 = square_x1 + side_length
        square_y2 = square_y1 + side_length

        # --- 5. Crop directly from the original image (no canvas needed) ---
        cropped_image_np = image_np[square_y1:square_y2, square_x1:square_x2, :]
        cropped_mask_np = mask_np[square_y1:square_y2, square_x1:square_x2]

        # --- 6. Convert to Tensor for Resizing ---
        image_for_resize = torch.from_numpy(cropped_image_np).unsqueeze(0)
        mask_for_resize = torch.from_numpy(cropped_mask_np).unsqueeze(0)

        # --- 7. Resize ---
        img_chw = image_for_resize.permute(0, 3, 1, 2)
        resized_image = torch.nn.functional.interpolate(
            img_chw, size=(target_size, target_size), mode='bicubic', align_corners=False
        ).permute(0, 2, 3, 1)

        mask_chw = mask_for_resize.unsqueeze(1)
        resized_mask = torch.nn.functional.interpolate(
            mask_chw, size=(target_size, target_size), mode='nearest'
        ).squeeze(1)
        
        # --- 8. Create CROP_DATA with the new square coordinates ---
        crop_data = (square_x1, square_y1, square_x2, square_y2, original_width, original_height)
        
        print(f"✅ MidnightLook (Crop): Zoom-cropped to square [{square_x1}, {square_y1}, {square_x2}, {square_y2}] and resized.")

        return (resized_image, resized_mask, (crop_data,))

class MidnightLook_PasteAfterInpaint:
    """
    Pastes a resized (1:1 aspect ratio) inpainted image back to its original location,
    handling the square crop data correctly.
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
        # --- 1. Unpack CROP_DATA ---
        data = crop_data[0]
        x1, y1, x2, y2, original_width, original_height = data
        
        pasted_image_tensor = original_image.clone()
        
        # --- 2. Resize inpainted image back to original square crop size ---
        side_length = x2 - x1
        
        inpainted_chw = inpainted_image.permute(0, 3, 1, 2)
        resized_inpainted = torch.nn.functional.interpolate(
            inpainted_chw, size=(side_length, side_length), mode='bicubic', align_corners=False
        )
        
        # --- 3. Determine the slice of the inpainted image to paste ---
        src_x_start = max(0, -x1)
        src_y_start = max(0, -y1)
        src_x_end = side_length - max(0, x2 - original_width)
        src_y_end = side_length - max(0, y2 - original_height)

        # --- 4. Determine the destination on the original image ---
        dest_x_start = max(0, x1)
        dest_y_start = max(0, y1)
        dest_x_end = min(original_width, x2)
        dest_y_end = min(original_height, y2)

        # --- 5. Slice and Paste ---
        if src_x_end > src_x_start and src_y_end > src_y_start:
            inpainted_slice = resized_inpainted[:, :, src_y_start:src_y_end, src_x_start:src_x_end]
            inpainted_slice_hwc = inpainted_slice.permute(0, 2, 3, 1)
            pasted_image_tensor[0, dest_y_start:dest_y_end, dest_x_start:dest_x_end, :] = inpainted_slice_hwc[0]
        
        print(f"✅ MidnightLook (Paste): Pasted inpainted image back to square region [{x1}, {y1}, {x2}, {y2}]")

        return (pasted_image_tensor,)

