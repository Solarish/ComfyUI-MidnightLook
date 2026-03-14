# Project Development Report: ComfyUI-MidnightLook
**Date:** March 14, 2026
**Status:** In Active Development / Production Ready

---

## 1. Project Overview
**ComfyUI-MidnightLook** เป็นชุด Custom Nodes ประสิทธิภาพสูงสำหรับ ComfyUI ที่เน้นการประมวลผล Image-to-Text (VLM), การจัดการ Prompt ที่ซับซ้อน, และการทำ Image Refinement (Detailer/Upscale) โดยมีการออกแบบโครงสร้างพื้นฐานให้รองรับการทำงานทั้งบน Local Environment และ Serverless Architecture (เช่น RunPod)

---

## 2. Core Infrastructure (Docker & Linux)
โครงสร้างพื้นฐานถูกออกแบบมาให้ "พกพาได้" และ "ปรับขนาดได้" (Portable & Scalable)
- **Base Environment:** Ubuntu 22.04 + CUDA 12.4 + Python 3.12 (optimized for RTX 40/Ada Lovelace)
- **Persistent Storage:** ใช้ระบบ Volume Mounting เพื่อแยก Model และ Custom Nodes ออกจากไฟล์ระบบของ Container ช่วยให้การอัปเดต Image ทำได้รวดเร็วโดยไม่ต้องดาวน์โหลดโมเดลใหม่
- **RunPod Compatibility:** รองรับการทำงานในลักษณะ Serverless พร้อมระบบจัดการ Network Volume และ Symlinks ที่แข็งแกร่ง

---

## 3. Custom Nodes Ecosystem
โครงการประกอบด้วย Node กลุ่มหลักดังนี้:

### A. Vision Language Models (Qwen2.5-VL)
- **MidnightQwen25Load:** ระบบโหลดโมเดลอัจฉริยะที่รองรับการค้นหาแบบ Recursive และ Symlink Traversal (ล่าสุดได้แก้ปัญหา path case-sensitivity และ symlink discovery)
- **MidnightQwen25Run:** รองรับการทำ Batch Inference ของรูปภาพและการประมวลผลวิดีโอ (VLM)

### B. Prompt & Logic Engineering
- **Z-Image Prompt:** ระบบจัดการ Template Prompt ที่มีความยืดหยุ่นสูง
- **Preset Prompt:** ระบบ Preset ที่ช่วยให้เรียกใช้ Prompt ประจำได้รวดเร็วผ่าน `preset.json`
- **Loop Control:** Nodes สำหรับจัดการ Iterative Logic ใน workflow
- **Midnight TextBox:** ประมวลผลและส่งต่อข้อมูลข้อความ

### C. Advanced Image Processing
- **Midnight Detailer:** เครื่องมือ Refinement เฉพาะจุด
- **Iterative Upscale:** ระบบขยายภาพแบบแบ่งเป็นรอบๆ เพื่อรักษาความคมชัด
- **MediaPipe/DeepFace Crop:** ระบบตรวจจับใบหน้าและอุปกรณต่างๆ เพื่อการ Crop ภาพที่แม่นยำสำหรับการทำ LoRA Test หรือ Inpaint

---

## 4. Key Technical Improvements (Recent)
ในช่วงการพัฒนาที่ผ่านมา มีการแก้ไขเชิงเทคนิคที่สำคัญ:
1.  **Robust Model Discovery:** 
    - แก้ปัญหา "Value not in list" บน RunPod โดยเพิ่ม `followlinks=True` ในระบบ discovery
    - จัดการ Path Separators (`/` vs `\`) ให้เป็นกลาง (Cross-platform)
    - ระบบ Auto-Drill Down เพื่อค้นหาไฟล์ `config.json` ในระดับลึก
2.  **Case-Sensitivity Fix:** ทำให้ระบบโหลดโมเดลรองรับทั้งโฟลเดอร์ชื่อ `vlm` และ `VLM` ในระบบ Linux
3.  **UI/UX Enhancements:** ปรับปรุง Error Logging ใน ComfyUI Console ให้ชัดเจน ระบุตำแหน่ง Path ที่ผิดพลาดได้ทันที

---

## 5. Deployment & Automation
- **Model Downloader:** สคริปต์ `download_models.py` สำหรับเตรียมโมเดลมาตรฐาน (Z-Image Base/Turbo, Qwen encoders)
- **Automation Scripts:** สคริปต์สำหรับการล้าง Docker Cache และการตรวจสอบสิทธิ์ใน S3 เพื่อการทำ Production Scaling

---

## 6. Next Steps
- พัฒนาระบบ **Auto-VRAM Management** สำหรับโมเดล VLM ขนาดใหญ่
- เพิ่มการสนับสนุน **SAM 2.1** ในโหนดการ Crop อัตโนมัติ
- ขยาย Preset Library สำหรับงาน Photography และ Digital Art เฉพาะทาง

---
**Report by:** Antigravity (Advanced Coding Agent)
*Reference files: nodes/*.py, .infrastructure/infrastructure.md, requirements.txt*
