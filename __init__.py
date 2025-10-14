# ComfyUI/custom_nodes/ComfyUI-MidnightLook/__init__.py

# Import class จากไฟล์ StringToBBOX.py
from .StringToBBOX import MidnightLook_StringToBBOX

# สร้าง Dictionary เพื่อ map ชื่อคลาสกับตัวคลาส
NODE_CLASS_MAPPINGS = {
    "MidnightLook_StringToBBOX": MidnightLook_StringToBBOX
}

# สร้าง Dictionary เพื่อ map ชื่อคลาสกับชื่อที่จะแสดงผลใน UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "MidnightLook_StringToBBOX": "Crop data to BBOX"
}

# พิมพ์ข้อความยืนยันเมื่อโหลดสำเร็จ (optional แต่แนะนำ)
print("✅ MidnightLook: Loaded custom nodes.")