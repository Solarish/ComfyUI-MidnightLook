# Development & Infrastructure Report (MidnightLook)
**Current Status & Infrastructure Specification: 2026-03-18**

This report summarizes the current development state of the `ComfyUI-MidnightLook` pipeline and its underlying infrastructure.

---

## 1. Production Pipeline Hardening (V4)
The pipeline from Payment to Training and Generation is now stabilized with the following fixes:

- **Webhook Synchronization:** Resolved Stripe PaymentIntent vs. Checkout Session mismatch. Webhooks now listen for `payment_intent.succeeded`.
- **LoRA URL Extraction:** Implemented robust extraction of `output.storage_key` from RunPod, converting it into a full `R2_CUSTOM_DOMAIN` LoRA URL.
- **Credit Leak Prevention:** Implemented "Pre-dispatch DB Lock" to ensure job records are created before firing expensive API requests to RunPod.
- **Timeout Protection:** Added `maxDuration = 60` for all serverless routes on Vercel to prevent 504 Gateway Timeouts.

---

## 2. Infrastructure Specification (Docker/Runner)
Based on the current `.infrastructure/infrastructure.md` and `docker-compose.yml`:

- **Environment:**
  - **Base Image:** `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` (RTX 4070 / Ada Lovelace Compatible).
  - **Python:** 3.12 (Primary).
  - **CUDA:** 12.4.
- **Resource Limits:**
  - **Shared Memory:** 4GB.
  - **Memory Limit:** 24GB.
- **Critical GPU Drivers:** Requires NVIDIA Container Toolkit installed on the host.
- **Persistent Volumes:**
  - `/app/ComfyUI/models` -> External Model drive.
  - `/app/ComfyUI/custom_nodes` -> Workspace sync.

---

## 3. Custom Node Infrastructure Dependencies
Each node group requires specific environmental setups:

- **VLM (Qwen2.5-VL):** Requires `transformers`, `qwen_vl_utils`, and `flash-attention-2` (optional but recommended for speed).
- **Face Analysis:** Requires `deepface`, `mediapipe`, and `insightface` (if used by specific backends).
- **Segmentation:** Requires `segment-anything` and `sam2`.
- **Cloud Storage:** Requires `boto3` and valid `R2_` environment variables.

---

## 4. Maintenance Commands & Access
- **Update Types:** `npx supabase gen types typescript --project-id xhlfuoggntepqjxqwawd > utils/supabase/types.ts`
- **RunPod CLI:** `./bin/runpodctl` for GPU monitoring and serverless management.
- **Cloudflare R2:** `npx wrangler r2 object list midnightlook-uploads` to verify uploads.

---

## 5. Next Steps
- [ ] Finalize the "Build-Your-Own 4-Pack" generation logic in the frontend.
- [ ] Verify SAM2 mask accuracy on wide-angle portrait shots.
- [ ] Monitor RunPod Serverless cold start times for the VLM endpoint.

**Report Generated:** 2026-03-18 09:53 (Local Time)
