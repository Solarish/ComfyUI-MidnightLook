# ComfyUI/custom_nodes/ComfyUI-MidnightLook/__init__.py

# Import classes from their respective files
from .StringToBBOX import MidnightLook_StringToBBOX
from .CropDataToBBox import MidnightLook_CropDATAToBBOX
from .InpaintUtils import MidnightLook_CropForInpaint, MidnightLook_PasteAfterInpaint
from .AnyToString import MidnightLook_AnyToString
from .LatentSizePresets import MidnightLook_LatentSizePresets
from .LoadImgByURL import MidnightLook_LoadImageByURL

# A dictionary that maps class names to class objects
NODE_CLASS_MAPPINGS = {
    "MidnightLook_StringToBBOX": MidnightLook_StringToBBOX,
    "MidnightLook_CropDATAToBBOX": MidnightLook_CropDATAToBBOX,
    "MidnightLook_CropForInpaint": MidnightLook_CropForInpaint,
    "MidnightLook_PasteAfterInpaint": MidnightLook_PasteAfterInpaint,
    "MidnightLook_AnyToString": MidnightLook_AnyToString,
    "MidnightLook_LatentSizePresets": MidnightLook_LatentSizePresets, # <-- เพิ่ม Node ใหม่
    "MidnightLook_LoadImageByURL": MidnightLook_LoadImageByURL, # <-- เพิ่ม Node ใหม่
}

# A dictionary that maps class names to UI display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "MidnightLook_StringToBBOX": "String to BBox (ML)",
    "MidnightLook_CropDATAToBBOX": "Crop Data to BBox (ML)",
    "MidnightLook_CropForInpaint": "Crop For Inpaint (ML)",
    "MidnightLook_PasteAfterInpaint": "Paste After Inpaint (ML)",
    "MidnightLook_AnyToString": "Any to String (ML)",
    "MidnightLook_LatentSizePresets": "Latent Size Presets (ML)", # <-- เพิ่มชื่อที่แสดงผล
    "MidnightLook_LoadImageByURL": "Load Image By URL (ML)", # <-- เพิ่มชื่อที่แสดงผล
}

# Print a confirmation message when nodes are loaded
print("✅ MidnightLook: Custom nodes loaded successfully.")

