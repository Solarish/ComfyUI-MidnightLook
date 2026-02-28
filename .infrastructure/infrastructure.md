# Docker Infrastructure Summary

## 1. Base Image & Environment
- **Base Image:** `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` (รองรับ RTX 4070 / Ada Lovelace)
- **OS:** Ubuntu 22.04
- **Python Version:** 3.12 (ตั้งเป็น default)
- **CUDA Version:** 12.4

## 2. Core Components
- **Application:** ComfyUI (Latest from GitHub)
- **Dependencies:**
    - Torch ecosystem (จัดการผ่าน requirements.txt ของ ComfyUI + ติดตั้ง `torchaudio`, `xformers` เพิ่มเติมเพื่อให้ version ตรงกัน)
    - AI Libraries: `diffusers`, `transformers`, `accelerate`, `safetensors`, `mediapipe`, `deepface`, `ultralytics`
    - Custom Node Support: `segment-anything`, `sam2`, `boto3`, `torchmetrics`

## 3. Models & Components
### Downloaded Models (via `download_models.py`)
- **Diffusion Models:**
    - `Z-Image Turbo` (BF16) - `models/diffusion_models/z_image_turbo_bf16.safetensors`
    - `Z-Image Base` (6B, BF16) - `models/diffusion_models/z_image_6b_bf16.safetensors`
- **Text Encoders:**
    - `Qwen 3.4B` - `models/text_encoders/qwen_3_4b.safetensors`
- **VAE:**
    - `AE` (Autoencoder) - `models/vae/ae.safetensors`

### Installed/Integrated Components
- **SAM (Segment Anything Model):**
    - `segment-anything` (Library)
    - `sam2` (via git from Facebook Research)
- **Face Analysis:**
    - `DeepFace` (Library + Weights caching at `/root/.deepface/weights`)
    - `MediaPipe` (Library)
- **Others:**
    - `Ultralytics` (YOLO for object detection)

## 4. Docker Configuration (docker-compose.yml)
- **Service Name:** `comfyui`
- **Container Name:** `z-image-v1`
- **Ports:** `8188:8188` (ComfyUI Web Interface)
- **GPU Resources:** ใช้งาน NVIDIA GPU ทั้งหมดที่มี (`count: all`)
- **Memory Settings:**
    - Shared Memory (`shm_size`): 4GB
    - Memory Limit: 24GB
- **Volumes (Persistent Storage):**
    - `C:\Dev\Docker\models` -> `/app/ComfyUI/models` (เก็บ Model ภายนอก Container)
    - `C:\Dev\Docker\custom_nodes` -> `/app/ComfyUI/custom_nodes` (เก็บ Custom Nodes ภายนอก Container)
- **Startup Command:**
    - `python3 main.py --listen 0.0.0.0 --lowvram --fp8_e4m3fn-text-enc`
    - มีการใช้ `LD_PRELOAD` กับ `libtcmalloc.so.4` เพื่อช่วยจัดการ Memory Allocation

## 5. Build Process (Dockerfile)
1. **System Setup:** ติดตั้ง System Dependencies (`git`, `curl`, `libgl1`, `build-essential`)
2. **Python Environment:** ติดตั้งและตั้งค่า Python 3.12 เป็นค่าเริ่มต้น
3. **ComfyUI Setup:** Clone ComfyUI repository ล่าสุด
4. **Dependency Installation:** ติดตั้ง Python dependencies (ทั้งของ ComfyUI และ library เสริมสำหรับ Z-Image / Custom Nodes)
5. **Directory Setup:** สร้างโฟลเดอร์สำหรับ Models เพื่อเตรียมพร้อมสำหรับ Volume Mounting
6. **Utility Scripts:** Copy สคริปต์ `download_models.py` เข้าไปใน image
