import torch

# สร้าง list ของ data type ที่ใช้บ่อยๆ ใน ComfyUI
# การทำแบบนี้จะช่วยให้ Validator ของ ComfyUI เข้าใจและยอมรับการเชื่อมต่อได้ดีขึ้น
COMFY_ANY_TYPE = ["IMAGE", "MASK", "LATENT", "MODEL", "CONDITIONING", "CROP_DATA", "*"]

class MidnightLook_AnyToString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # เปลี่ยนจาก "*" ธรรมดามาเป็น list ที่เรากำหนดไว้
                "input_data": (COMFY_ANY_TYPE,),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "any_to_string"
    CATEGORY = "MidnightLook/Utils"

    def any_to_string(self, input_data):
        # ตรวจสอบว่าข้อมูลเป็น Tensor หรือไม่
        if isinstance(input_data, torch.Tensor):
            # ถ้าเป็น Tensor ให้แสดงข้อมูล shape และค่าบางส่วน จะได้อ่านง่ายขึ้น
            output_string = f"Tensor Shape: {input_data.shape}, DType: {input_data.dtype}, Device: {input_data.device}"
        else:
            # ถ้าเป็นข้อมูลชนิดอื่น ก็แปลงเป็น string ตรงๆ
            output_string = str(input_data)
        
        # แสดงผลใน Console เพื่อช่วยในการดีบัก
        print(f"✅ MidnightLook (AnyToString): Converted data of type '{type(input_data)}' to string.")
        print(f"    Value: {output_string}")
        
        return (output_string,)

