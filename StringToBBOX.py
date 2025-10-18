# ComfyUI/custom_nodes/ComfyUI-MidnightLook/StringToBBOX.py

import torch
import ast # ใช้สำหรับแปลง string ที่มีหน้าตาเหมือน code python ให้เป็น object จริงๆ อย่างปลอดภัย

class MidnightLook_StringToBBOX:
    """
    A custom node to parse a string representation of a tuple containing coordinates 
    and a bounding box, and return the bounding box as a BBOX tensor.
    
    Input format example: "((322, 322), (330, 174, 652, 496))"
    Output BBOX: [330, 174, 652, 496]sdf
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data_string": ("STRING", {
                    "multiline": False,
                    "default": "((0, 0), (0, 0, 512, 512))"
                }),
            }
        }

    RETURN_TYPES = ("BBOX",)
    FUNCTION = "process"
    CATEGORY = "MidnightLook/Utils" # จัดหมวดหมู่ให้ Node ของเรา

    def process(self, data_string: str):
        try:
            # ใช้ ast.literal_eval() เพื่อแปลง string เป็น Python object อย่างปลอดภัย
            # มันปลอดภัยกว่าการใช้ eval() มาก
            parsed_data = ast.literal_eval(data_string)

            # ตรวจสอบโครงสร้างข้อมูลที่ได้รับ
            if not isinstance(parsed_data, tuple) or len(parsed_data) != 2:
                raise ValueError("Input string must be a tuple of two elements.")
                
            bbox_tuple = parsed_data[1]
            
            if not isinstance(bbox_tuple, tuple) or len(bbox_tuple) != 4:
                raise ValueError("The second element must be a tuple of four numbers (left, top, right, bottom).")

            # แปลง tuple ของ BBOX ให้เป็น torch.Tensor ที่ ComfyUI ต้องการ
            # BBOX ใน ComfyUI มักจะอยู่ในรูปแบบ list ของ tensor (เผื่อสำหรับ batch processing)
            # Tensor ควรเป็น dtype=int64 หรือ float32 ก็ได้ ในที่นี้ใช้ int64
            left, top, right, bottom = bbox_tuple
            bbox_tensor = torch.tensor([left, top, right, bottom], dtype=torch.int64)
            
            # ComfyUI คาดหวัง output ที่เป็น tuple ซึ่งสมาชิกตัวแรกคือ list ของ tensor
            return ([bbox_tensor], )

        except (ValueError, SyntaxError) as e:
            # หาก string ที่รับมามี format ไม่ถูกต้อง ให้แสดง error ใน console
            # และ return BBOX ค่า default เพื่อไม่ให้ workflow พัง
            print(f"ERROR: MidnightLook_StringToBBOX - Invalid input string format: {e}")
            print(f"    Input was: '{data_string}'")
            default_bbox = torch.tensor([0, 0, 512, 512], dtype=torch.int64)
            return ([default_bbox], )