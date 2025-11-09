import torch
from PIL import Image
from io import BytesIO
import numpy as np
import boto3
import time

class MidnightLook_UploadToR2:
    """
    Upload batch images from ComfyUI to Cloudflare R2 (or S3 compatible).

    INPUT:
        images        : ComfyUI IMAGE tensor [B, H, W, C] (float32, 0-1)
        bucket        : ชื่อ bucket บน R2
        endpoint_url  : S3 endpoint ของ R2 (เช่น https://<account_id>.r2.cloudflarestorage.com)
        access_key    : Access Key ID
        secret_key    : Secret Key
        job_id        : ไว้ใช้ grouping ไฟล์ในโฟลเดอร์ย่อย
        prefix        : โฟลเดอร์ root ใน bucket (default: "midnightlook")
        public_base_url : base URL สำหรับให้ user เข้าถึงไฟล์ (เช่น https://cdn.example.com)

    OUTPUT:
        keys : list[str]  - path ของไฟล์ใน bucket (เช่น "midnightlook/job123/ML_job123_1_123456789.jpg")
        urls : list[str]  - URL ที่พร้อมนำไปใช้ (public_base_url หรือ fallback จาก endpoint_url)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),  # batch image [B,H,W,C]
                "bucket": ("STRING", {"multiline": False}),
                "endpoint_url": ("STRING", {"multiline": False}),
                "access_key": ("STRING", {"multiline": False}),
                "secret_key": ("STRING", {"multiline": True}),
                "job_id": ("STRING", {"multiline": False, "default": "unknown"}),
            },
            "optional": {
                "prefix": ("STRING", {"multiline": False, "default": "midnightlook"}),
                # ใช้สำหรับสร้าง URL สาธารณะ เช่น https://cdn.my-domain.com
                "public_base_url": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")   # keys, urls
    RETURN_NAMES = ("keys", "urls")
    FUNCTION = "upload"
    CATEGORY = "MidnightLook/Image"

    # บอก ComfyUI ว่า output ทั้งสองช่องเป็น list ของ STRING
    OUTPUT_IS_LIST = (True, True)

    def upload(
        self,
        images,
        bucket,
        endpoint_url,
        access_key,
        secret_key,
        job_id,
        prefix="midnightlook",
        public_base_url="",
    ):
        try:
            # ตรวจ shape คร่าว ๆ กันพลาด
            if images.dim() != 4:
                raise ValueError(f"Expected IMAGE with shape [B,H,W,C], got {tuple(images.shape)}")

            if images.shape[-1] not in (3, 4, 1):
                print(f"[WARN] Unexpected channel count in images: {images.shape[-1]}")

            # สร้าง S3 client (ใช้กับ R2)
            s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="auto",
            )

            uploaded_keys = []
            uploaded_urls = []

            # loop ทีละภาพใน batch
            for i, img_tensor in enumerate(images):
                # img_tensor: [H,W,C], float32 (0-1)
                img_np = img_tensor.cpu().numpy()

                # กัน value หลุด range แล้วแปลงเป็น uint8
                img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

                # ถ้า C=1 ให้แปลงเป็น RGB
                if img_np.ndim == 3 and img_np.shape[2] == 1:
                    img_np = np.repeat(img_np, 3, axis=2)

                # ถ้า C=4 ตัด alpha ทิ้งให้เหลือ 3 channel
                if img_np.ndim == 3 and img_np.shape[2] == 4:
                    img_np = img_np[:, :, :3]

                img_pil = Image.fromarray(img_np, "RGB")

                buffer = BytesIO()
                img_pil.save(buffer, format="JPEG", quality=95)
                buffer.seek(0)

                timestamp = int(time.time() * 1000)
                file_key = f"{prefix}/{job_id}/ML_{job_id}_{i+1}_{timestamp}.jpg"

                # อัปโหลดไป R2
                s3_client.upload_fileobj(
                    buffer,
                    bucket,
                    file_key,
                    ExtraArgs={"ContentType": "image/jpeg"},
                )

                uploaded_keys.append(file_key)

                # สร้าง URL
                if public_base_url.strip():
                    base = public_base_url.rstrip("/")
                    url = f"{base}/{file_key.lstrip('/')}"
                else:
                    # fallback: ใช้ endpoint_url + bucket + key
                    # (กรณีนี้ต้องตั้ง permission ของ bucket ให้เข้าถึงได้เอง)
                    base = endpoint_url.rstrip("/")
                    url = f"{base}/{bucket}/{file_key}"

                uploaded_urls.append(url)

                print(f"✅ Uploaded to R2: {file_key} -> {url}")

            return (uploaded_keys, uploaded_urls)

        except Exception as e:
            print(f"ERROR: MidnightLook_UploadToR2 - Failed to upload batch images to R2: {e}")
            # คืนค่าเป็น list เพื่อให้ type ยังตรงกับ OUTPUT_IS_LIST
            return ([f"ERROR: {e}"], [""])

# --- mappings สำหรับ ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "MidnightLook_UploadToR2": MidnightLook_UploadToR2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MidnightLook_UploadToR2": "Upload Image to R2 (ML)"
}
