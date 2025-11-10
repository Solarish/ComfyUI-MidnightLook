import torch
from PIL import Image
from io import BytesIO
import numpy as np
import boto3
import time
import os

class MidnightLook_UploadToR2:
    """
    (v3) อัปโหลด 'รูปเดียว' (Single Image) ไปยัง R2 โดยใช้ billing_id เป็น Path
    
    INPUT:
        image         : ComfyUI IMAGE tensor [1, H, W, C] (float32, 0-1)
        bucket        : (String) R2 Bucket Name
        endpoint_url  : (String) S3 Endpoint URL (e.g., https://<account_id>.r2.cloudflarestorage.com)
        access_key    : (String) R2 Access Key
        secret_key    : (String) R2 Secret Key
        billing_id    : (String) ID การจ่ายเงิน (Parent ID)
        user_email    : (String) Email ของ User
        file_index    : (Int) ลำดับของรูปภาพ (เช่น 1, 2, 3...)
        prefix        : (String) โฟลเดอร์ Root (e.g., "outputs")

    OUTPUT:
        url : (String) - URL ของไฟล์ที่อัปโหลด (e.g., "https://cdn.example.com/outputs/user@email.com/billing123/ML_1_123456.jpg")
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",), # (FIX) รับทีละรูป (Batch size = 1)
                "bucket": ("STRING", {"multiline": False}),
                "endpoint_url": ("STRING", {"multiline": False}),
                "access_key": ("STRING", {"multiline": False}),
                "secret_key": ("STRING", {"multiline": True}),
                "billing_id": ("STRING", {"multiline": False, "default": "unknown"}),
                "user_email": ("STRING", {"multiline": False, "default": "unknown"}),
                "file_index": ("INT", {"default": 1, "min": 1, "max": 999}),
                "public_base_url": ("STRING", {"multiline": False, "default": ""}),
                "prefix": ("STRING", {"multiline": False, "default": "outputs"}),
            }
        }

    # (FIX) คืนค่าเป็น URL (String) เดียว
    RETURN_TYPES = ("STRING",) 
    FUNCTION = "upload_image"
    CATEGORY = "MidnightLook/Image"
    
    # (FIX) Output เป็น 1 URL (ไม่ใช่ List)
    OUTPUT_NODE = True 

    def upload_image(self, image, bucket, endpoint_url, access_key, secret_key, billing_id, user_email, file_index, public_base_url, prefix="outputs"):
        try:
            # 1. เชื่อมต่อ S3 Client (R2)
            s3_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name='auto' 
            )

            # 2. แปลง Tensor (1, H, W, C) -> (H, W, C) -> PIL Image
            img_tensor = image[0] # ดึงรูปเดียวออกมา
            img_np = img_tensor.cpu().numpy() 
            img_pil = Image.fromarray(np.clip(img_np * 255.0, 0, 255).astype(np.uint8), 'RGB')

            # 3. สร้าง Buffer ใน Memory
            buffer = BytesIO()
            img_pil.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)
            
            # 4. สร้างชื่อไฟล์ (FIX v17)
            # ใช้ (email)-(billingID) เป็น Folder หลัก
            timestamp = int(time.time() * 1000)
            
            # (FIX v17) สร้าง Path ใหม่ตามที่คุณขอ: [email]-[billingID]/[filename]
            folder_path = f"{prefix}/{user_email}-{billing_id}"
            file_name = f"ML_{file_index}_{timestamp}.jpg"
            file_key = f"{folder_path}/{file_name}"

            # 5. อัปโหลด
            s3_client.upload_fileobj(
                buffer,
                bucket,
                file_key,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )

            # 6. สร้าง Public URL
            if not public_base_url.strip():
                 raise Exception("public_base_url is required but was not provided.")
                 
            base = public_base_url.rstrip("/")
            url = f"{base}/{file_key.lstrip('/')}"
            
            print(f"✅ MidnightLook (UploadToR2): Uploaded image {file_index} to {url}")
            
            # 7. คืนค่าเป็น URL (String) เดียว
            return (url,)

        except Exception as e:
            print(f"ERROR: MidnightLook_UploadToR2 - Failed to upload image to R2: {e}")
            return ("ERROR: {e}",)
            
            
# --- (FIX) ลบ LoadImageByURL ออก (แยกไฟล์) ---


# --- mappings สำหรับ ComfyUI (เหลือแค่ Node เดียว) ---
NODE_CLASS_MAPPINGS = {
    "MidnightLook_UploadToR2": MidnightLook_UploadToR2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MidnightLook_UploadToR2": "Upload Image to R2 (ML)"
}