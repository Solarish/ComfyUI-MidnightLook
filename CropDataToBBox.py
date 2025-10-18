import torch

class MidnightLook_CropDATAToBBOX:
    """
    A custom node to convert CROP_DATA (a tuple containing crop coordinates) 
    into a BBOX tensor.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "crop_data": ("CROP_DATA",),
            },
        }

    RETURN_TYPES = ("BBOX",)
    FUNCTION = "process"
    CATEGORY = "MidnightLook/Utils"

    def process(self, crop_data):
        # Based on InpaintUtils.py, crop_data is a tuple containing another tuple: ((x1, y1, x2, y2, ...),)
        # We need the inner tuple.
        crop_info = crop_data[0]
 # Ensure crop_info has at least 4 elements before attempting to unpack
        if not isinstance(crop_info, tuple) or len(crop_info) < 4:
            raise ValueError("Invalid crop_data format. Expected a tuple with at least 4 elements for bounding box.")

        # Extract the bounding box coordinates (left, top, right, bottom)
        left, top, right, bottom = crop_info[0], crop_info[1], crop_info[2], crop_info[3]

        bbox_tensor = torch.tensor([left, top, right, bottom], dtype=torch.int64)
        
        # ComfyUI expects the output to be a tuple containing a list of tensors
        return ([bbox_tensor],)
