# MidnightLook Nodes Documentation

## 1. Midnight Qwen2.5 Run (VLM Analysis)
โหนดสำหรับวิเคราะห์รูปภาพและสร้างคำบรรยายด้วยโมเดล Qwen2.5-VL

### Input Parameters
*   **system_text**: บทบาทหน้าที่ของ AI (System Prompt)
    *   *ค่าแนะนำ*: "You are an expert in analyzing human physical appearance..." (เน้นอธิบายรูปร่างหน้าตา)
*   **text**: คำสั่งเฉพาะเจาะจง (User Prompt)
    *   *ค่าแนะนำ*: "Describe the physical appearance of the person in the image in detail."
*   **max_new_tokens**: จำนวนคำตอบสูงสุด (default: 512)
*   **min_pixels / max_pixels**: ความละเอียดภาพที่ส่งให้โมเดล (default: 256 - 1280)
*   **description_input** (Optional): ข้อความเสริมที่จะถูกนำไปต่อท้าย Trigger Word ใน Prompt

---

## 2. Z-Image Prompt Prep (ML)
โหนดช่วยสร้าง Prompt สำหรับ Z-Image Turbo แบบง่ายๆ โดยมี Preset ให้เลือก

### Input Categories
โหนดนี้แบ่งหมวดหมู่การใส่ Prompt เป็น 8 ช่อง:
1.  **Trigger Word**: คำเรียก LoRA ตลอดจนชื่อเฉพาะ
2.  **Subject**: ประธานของภาพ (เช่น "A 25-year-old woman")
3.  **Outfit**: ชุดแต่งกาย
4.  **Pose**: ท่าทาง
5.  **Background**: ฉากหลัง
6.  **Lighting**: แสงและบรรยากาศ
7.  **Style**: สไตล์ภาพและกล้อง (เช่น "Film (Leica M6)", "Studio (8K)")
8.  **Text Rendering**: ข้อความที่ต้องการให้ปรากฏในภาพ

ทุกช่องจะมี **Dropdown (Preset)** ให้เลือกแบบด่วน หรือจะพิมพ์เองในช่อง **Custom** ก็ได้

---

## 3. Face Identity Score (DeepFace Verify)
โหนดสำหรับตรวจสอบความเหมือนของหน้าตาบุคคลโดยใช้โมเดล DeepFace (Commercial Friendly)

> **Features**: ใช้โมเดลระดับโลกอย่าง FaceNet หรือ SFace แม่นยำกว่า DINOv2 ในการระบุตัวตน

### Input Parameters
*   **image1**: รูปต้นฉบับ (Reference)
*   **image2**: รูปที่ AI สร้างมา (Generated)
*   **model_name**: รุ่นของโมเดล
    *   `FaceNet512`: แม่นยำสูง (แนะนำ)
    *   `FaceNet`: รุ่นมาตรฐาน
    *   `SFace`: เร็วและเบา
    *   `OpenFace`: ทางเลือกเพิ่มเติม
*   **threshold**: ค่าความห่างสูงสุดที่ยอมรับได้ (default: 0.30)
    *   ⚠️ **ค่ายิ่งน้อย = ยิ่งเข้มงวด** (ต่างจากคะแนนเปอร์เซ็นต์ทั่วไป!)
    *   **Facenet512 + Cosine**: แนะนำ `0.30` (ถ้า dist ≤ 0.30 = คนเดียวกัน)
    *   ตั้ง `0.20` = เข้มงวดมาก / ตั้ง `0.40` = ผ่อนปรน
    *   info_text จะแสดงทั้ง Threshold ที่คุณตั้ง และ Recommended ของแต่ละโมเดลให้อัตโนมัติ

### Outputs
*   **distance**: ค่าความห่าง (Distance Score)
*   **is_match**: ผ่านเกณฑ์หรือไม่ (True/False)
*   **verified_image**: รูปที่ผ่านเกณฑ์ (ถ้าไม่ผ่านจะเป็นสีดำ)
*   **info_text**: ข้อความสรุปผลพร้อม Icon

