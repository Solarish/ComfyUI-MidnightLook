# ComfyUI/custom_nodes/ComfyUI-MidnightLook/__init__.py

# Import classes from their respective files
from .StringToBBOX import MidnightLook_StringToBBOX
from .InpaintUtils import MidnightLook_CropForInpaint, MidnightLook_PasteAfterInpaint

# A dictionary that maps class names to class objects
NODE_CLASS_MAPPINGS = {
    "MidnightLook_StringToBBOX": MidnightLook_StringToBBOX,
    "MidnightLook_CropForInpaint": MidnightLook_CropForInpaint,
    "MidnightLook_PasteAfterInpaint": MidnightLook_PasteAfterInpaint,
}

# A dictionary that maps class names to UI display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "MidnightLook_StringToBBOX": "Crop data to BBOX",
    "MidnightLook_CropForInpaint": "Crop For Inpaint",
    "MidnightLook_PasteAfterInpaint": "Paste After Inpaint",
}

# Print a confirmation message when nodes are loaded
print("✅ MidnightLook: Custom nodes loaded successfully.")
