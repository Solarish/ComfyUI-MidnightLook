import torch
import numpy as np
import scipy.ndimage
import comfy.utils
import nodes
import folder_paths
import os

# Register missing model folders for dropdown population
if "sams" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["sams"] = ([os.path.join(folder_paths.models_dir, "sams"), os.path.join(folder_paths.models_dir, "sam")], folder_paths.supported_pt_extensions)
else:
    # Ensure "sam" is checked along with "sams"
    sams_paths = folder_paths.folder_names_and_paths["sams"][0]
    sam_path = os.path.join(folder_paths.models_dir, "sam")
    if sam_path not in sams_paths:
        sams_paths.append(sam_path)

if "grounding-dino" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["grounding-dino"] = ([os.path.join(folder_paths.models_dir, "grounding-dino")], folder_paths.supported_pt_extensions)

_dino_cache = {}
_sam2_cache = {}

class SAM2LoaderNode:
    """
    Loads a SAM2 model from `models/sams` and performs text-prompted or center-focused segmentation.
    Outputs a bounding box and a mask.
    """
    @classmethod
    def INPUT_TYPES(cls):
        sam_models = folder_paths.get_filename_list("sams")
        if not sam_models:
             sam_models = ["sam2_hiera_large.pt", "model.safetensors"]
             
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": False, "default": "face"}),
                "sam2_model_name": (sam_models, ),
                "dino_model_dir": ("STRING", {"multiline": False, "default": "models/grounding-dino/grounding-dino-base"}),
                "box_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "text_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("BBOX", "MASK")
    RETURN_NAMES = ("bbox", "mask")
    FUNCTION = "process"
    CATEGORY = "MidnightLook/Detailer"

    def process(self, image, prompt, sam2_model_name, dino_model_dir, box_threshold, text_threshold):
        global _dino_cache, _sam2_cache
        import folder_paths
        import os
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # image shape: [B, H, W, C]
        b, h, w, c = image.shape
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        
        # 1. Load GroundingDINO Model via Transformers
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError:
            print("⚠️ SAM2Loader: transformers library not found. Returning empty mask.")
            return ([torch.tensor([0, 0, w, h], dtype=torch.int64)], torch.zeros((1, h, w), dtype=torch.float32))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dino_model_dir = dino_model_dir.strip()
        
        # Robust Path Resolution
        dino_path = None
        
        # 1. Check if it's an absolute path
        if os.path.isabs(dino_model_dir) and os.path.exists(dino_model_dir):
            dino_path = dino_model_dir
        # 2. Check if it's relative to ComfyUI base path
        elif os.path.exists(os.path.join(folder_paths.base_path, dino_model_dir)):
            dino_path = os.path.join(folder_paths.base_path, dino_model_dir)
        else:
            # 3. Search inside ComfyUI's registered grounding-dino folders
            clean_name = dino_model_dir.replace(" ", "") # Fix accidental spaces
            for search_dir in folder_paths.get_folder_paths("grounding-dino"):
                # Check if the folder itself contains config.json (user placed files directly in models/grounding-dino)
                if os.path.exists(os.path.join(search_dir, "config.json")):
                    dino_path = search_dir
                    break
                # Check if the input is a subfolder
                elif os.path.exists(os.path.join(search_dir, dino_model_dir)):
                    dino_path = os.path.join(search_dir, dino_model_dir)
                    break
                # Check if the input is a file name inside the folder
                elif os.path.exists(os.path.join(search_dir, clean_name)):
                    dino_path = os.path.join(search_dir, clean_name)
                    break

        # Fallback if nothing found
        if not dino_path:
             dino_path = dino_model_dir
             
        # If the user accidentally points directly to the model file, use its parent directory instead
        if os.path.isfile(dino_path):
             dino_path = os.path.dirname(dino_path)
             
        if not os.path.exists(dino_path):
             print(f"⚠️ SAM2Loader: GroundingDINO directory not found at '{dino_path}'")
             return ([torch.tensor([0, 0, w, h], dtype=torch.int64)], torch.zeros((1, h, w), dtype=torch.float32))
        
        if dino_path in _dino_cache:
            processor, dino_model = _dino_cache[dino_path]
            print(f"⚡ SAM2Loader: Using cached GroundingDINO from {dino_path}")
        else:
            print(f"🔄 SAM2Loader: Loading GroundingDINO locally from {dino_path}")
            try:
                processor = AutoProcessor.from_pretrained(dino_path, local_files_only=True, use_fast=False)
                dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_path, local_files_only=True).to(device)
                _dino_cache[dino_path] = (processor, dino_model)
            except Exception as e:
                print(f"⚠️ SAM2Loader Error loading transformers GroundingDINO: {e}")
                import traceback
                traceback.print_exc()
                return ([torch.tensor([0, 0, w, h], dtype=torch.int64)], torch.zeros((1, h, w), dtype=torch.float32))

        # 2. Predict BBox with GroundingDINO
        from PIL import Image
        pil_image = Image.fromarray(img_np)
        
        prompt = prompt.lower().strip()
        if not prompt.endswith("."):
            prompt = prompt + "."
            
        print(f"DEBUG: Processing DINO Prompt: '{prompt}'")
            
        try:
            inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = dino_model(**inputs)

            # In ComfyUI, inputs like box_threshold might come in as primitive types/strings
            bt = float(box_threshold)
            tt = float(text_threshold)

            print(f"DEBUG: Applying GroundingDINO thresholds - Box: {bt}, Text: {tt}")
            
            # Print the literal max score from the logits to see what the model is predicting
            max_score = outputs.logits.sigmoid().max().item()
            print(f"DEBUG: Max DINO Logit Sigmoid Score in the entire image is: {max_score:.4f}")

            # Workaround for HuggingFace Transformers version differences on keyword arguments
            try:
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=bt,
                    text_threshold=tt,
                    target_sizes=[pil_image.size[::-1]] # (height, width)
                )[0]
            except Exception as inner_e:
                print(f"⚠️ Warning: post_process with thresholds failed ({inner_e}). Falling back to manual logit filtering.")
                # Since processor rejects the keywords, we might need to change it on the fly or just accept the defaults, 
                # but transformers 4.40+ hardcodes the threshold filter inside post_process if not provided.
                
                # We will manually threshold the raw logits before calling post process
                # Find valid logits that exceed our threshold
                logits = outputs.logits.sigmoid()
                
                # By passing a very low threshold we force post_process to keep the results.
                # Unfortunately post_process doesn't let us pass it, so we mock the pre-processor logic or set the class attribute
                if hasattr(processor, "box_threshold"):
                    processor.box_threshold = 0.001
                if hasattr(processor, "text_threshold"):
                    processor.text_threshold = 0.001
                    
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    target_sizes=[pil_image.size[::-1]]
                )[0]
                
            # GroundingDINO outputs a batch of results. We only sent 1 image.
            boxes = results["boxes"]
            scores = results["scores"]
            
        except Exception as e:
            print(f"⚠️ SAM2Loader DINO Inference Error: {e}")
            import traceback
            traceback.print_exc()
            return ([torch.tensor([0, 0, w, h], dtype=torch.int64)], torch.zeros((1, h, w), dtype=torch.float32))
        
        print(f"DEBUG: Found {len(scores)} total objects before filtering.")
        if len(scores) > 0:
             print(f"DEBUG: Max score is {scores.max().item():.4f}, threshold is {bt}")
        
        # Apply manual threshold filtering using PyTorch indexing
        # Now we apply our user-provided bounds filtering.
        valid_indices = scores > bt
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        
        print(f"DEBUG: {len(scores)} objects remained after thresholding.")
        
        if len(boxes) == 0:
            print(f"⚠️ SAM2Loader: GroundingDINO found no objects for prompt '{prompt}' above box_threshold {bt}")
            # Fallback BBOX is the whole image so the pipeline doesn't violently break
            return ([torch.tensor([0, 0, w, h], dtype=torch.int64)], torch.zeros((1, h, w), dtype=torch.float32))
            
        # Get highest confidence box
        best_box_idx = scores.argmax()
        best_box = boxes[best_box_idx] # [xmin, ymin, xmax, ymax]
        
        x1, y1, x2, y2 = [int(v.item()) for v in best_box]
        
        # Clamp bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        bbox_tensor = torch.tensor([x1, y1, x2, y2], dtype=torch.int64)
        print(f"✅ SAM2Loader DINO BBox for '{prompt}': {bbox_tensor.tolist()}")

        # 3. Load SAM2 Model
        sam_path = folder_paths.get_full_path("sams", sam2_model_name)
        if hasattr(folder_paths, "get_folder_paths"):
            if sam_path is None:
               sam_paths = folder_paths.get_folder_paths("sam")
               if sam_paths:
                   for fp in sam_paths:
                       test_path = os.path.join(fp, sam2_model_name)
                       if os.path.exists(test_path):
                           sam_path = test_path
                           break
        
        if sam_path is None:
             base_path = folder_paths.models_dir
             test_path = os.path.join(base_path, "sam", "model.safetensors")
             if os.path.exists(test_path):
                 sam_path = test_path
                 sam2_model_name = "model.safetensors"
             else:
                 test_path = os.path.join(base_path, "sams", sam2_model_name)
                 if os.path.exists(test_path):
                     sam_path = test_path

        if not sam_path or not os.path.exists(sam_path):
            print(f"⚠️ SAM2Loader: Could not find model {sam2_model_name}")
            return ([bbox_tensor], torch.zeros((1, h, w), dtype=torch.float32))
            
        if sam_path in _sam2_cache:
             predictor = _sam2_cache[sam_path]
             print(f"⚡ SAM2Loader: Using cached SAM2 from {sam_path}")
        else:
             print(f"🔄 SAM2Loader: Loading SAM2 from {sam_path}")
             if "large" in sam2_model_name:
                 model_cfg = "sam2_hiera_l.yaml"
             elif "base_plus" in sam2_model_name:
                 model_cfg = "sam2_hiera_b+.yaml"
             elif "small" in sam2_model_name:
                 model_cfg = "sam2_hiera_s.yaml"
             elif "tiny" in sam2_model_name:
                 model_cfg = "sam2_hiera_t.yaml"
             else:
                 model_cfg = "sam2_hiera_l.yaml"
     
             try:
                 # PyTorch 2.6+ defaults to weights_only=True which breaks SAM2 safetensors unpickling
                 # Also, SAM2 repository hardcodes torch.load() which doesn't support .safetensors out of the box.
                 original_load = torch.load
                 def safe_load(f, *args, **kwargs):
                     if isinstance(f, str) and f.endswith(".safetensors"):
                         import comfy.utils
                         sd = comfy.utils.load_torch_file(f)
                         # If it's a flat state dict, wrap it for SAM2's unpickler
                         if "model" not in sd:
                             return {"model": sd}
                         return sd
                     # Fallback to normal torch load for .pt files but disable weights_only to prevent Unpickler errors
                     kwargs['weights_only'] = False
                     return original_load(f, *args, **kwargs)
                 
                 torch.load = safe_load
                 sam2_model = build_sam2(model_cfg, sam_path, device=device)
                 torch.load = original_load
                 
                 predictor = SAM2ImagePredictor(sam2_model)
                 _sam2_cache[sam_path] = predictor
             except Exception as e:
                 # Restore in case of failure
                 if 'original_load' in locals():
                     torch.load = original_load
                 print(f"⚠️ SAM2Loader Error building SAM2: {e}")
                 import traceback
                 traceback.print_exc()
                 return ([bbox_tensor], torch.zeros((1, h, w), dtype=torch.float32))

        # 4. Predict SAM2 Mask using DINO BBox
        predictor.set_image(img_np)
        
        input_box = np.array([x1, y1, x2, y2])
        
        try:
            masks, scores, logits = predictor.predict(
                box=input_box,
                multimask_output=False,
            )
            
            # The mask returned by SAM2 is typically [num_masks, H, W] and boolean or float logits.
            # We take the best mask, convert to float 0.0/1.0, and reshape to [1, H, W]
            best_mask = masks[0] 
            
            print(f"DEBUG: SAM2 Mask shape: {best_mask.shape}, dtype: {best_mask.dtype}, min: {np.min(best_mask)}, max: {np.max(best_mask)}")
            
            # If it's already a float logit, we might need a threshold > 0.0
            # If it's boolean, we convert it.
            if best_mask.dtype == np.bool_:
                 mask_np_float = best_mask.astype(np.float32)
            else:
                 mask_np_float = (best_mask > 0.0).astype(np.float32)
                 
            out_mask = torch.from_numpy(mask_np_float).unsqueeze(0)
            
            print(f"DEBUG: Processed Mask shape: {out_mask.shape}, min: {out_mask.min()}, max: {out_mask.max()}")

        except Exception as e:
            print(f"⚠️ SAM2Loader Prediction Error: {e}")
            out_mask = torch.zeros((1, h, w), dtype=torch.float32)
            
        print(f"✅ SAM2Loader: Generated SAM2 mask from DINO BBox successfully.")
        return ([bbox_tensor], out_mask)



class MidnightDetailerNode:
    """
    Performs Crop and Stitch detailing on a region defined by a BBOX and MASK.
    Includes mask refinement (dilation/blur), cropping, inpainting via KSampler, and blending back.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 64}),
                "mask_expand": ("INT", {"default": 8, "min": -64, "max": 64}),
                "preset_prompt": (["None", "highly detailed face, skin pores, 8k resolution, masterpiece", "highly detailed eyes, beautiful catchlights, sharp focus", "highly detailed clothing texture, fabric weaves"], {"default": "None"}),
                "guide_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "bbox": ("BBOX",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "MidnightLook/Detailer"

    def process(self, image, mask, model, clip, vae, seed, steps, cfg, sampler_name, scheduler, denoise, target_size, mask_blur, mask_expand, preset_prompt, guide_prompt, bbox=None):
        b, h, w, c = image.shape
        
        # 1. Provide Fallback for BBOX Parsing
        if bbox is not None:
            bbox_data = bbox[0]
            x1, y1, x2, y2 = bbox_data.tolist()
        else:
            print("⚠️ MidnightDetailer: No valid BBOX provided. Returning original image.")
            return (image,)
            
        # Validate coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop_w, crop_h = x2 - x1, y2 - y1
        
        if crop_w <= 0 or crop_h <= 0:
            print("⚠️ MidnightDetailer: Invalid BBOX dimensions. Returning original image.")
            return (image,)
            
        # 2. Mask Refinement (Dilation/Erosion + Blur)
        # Convert mask to numpy for scipy operations
        mask_np = mask[0].cpu().numpy()
        
        # Dilate or Erode mask
        if mask_expand > 0:
            processed_mask = scipy.ndimage.grey_dilation(mask_np, size=(mask_expand, mask_expand))
        elif mask_expand < 0:
            processed_mask = scipy.ndimage.grey_erosion(mask_np, size=(-mask_expand, -mask_expand))
        else:
            processed_mask = mask_np
            
        # Gaussian blur for alpha blending feathering
        blurred_mask = scipy.ndimage.gaussian_filter(processed_mask, sigma=mask_blur)
        blurred_mask_tensor = torch.from_numpy(blurred_mask).float().unsqueeze(0) # [1, H, W]
        
        # 3. Crop Image & Mask
        # Add margin to crop box so expanded masks aren't cut off by the original tight bbox
        margin = max(0, mask_expand) + (mask_blur * 2) + 16
        
        # Square crop logic
        center_x, center_y = x1 + crop_w / 2.0, y1 + crop_h / 2.0
        side_len = max(crop_w, crop_h) + int(margin * 2)
        sq_x1, sq_y1 = int(center_x - side_len / 2.0), int(center_y - side_len / 2.0)
        sq_x2, sq_y2 = sq_x1 + side_len, sq_y1 + side_len
        
        # Pad if out of bounds
        img_np = image[0].cpu().numpy()
        padded_img = np.pad(img_np, ((max(0, -sq_y1), max(0, sq_y2 - h)), (max(0, -sq_x1), max(0, sq_x2 - w)), (0, 0)), mode='reflect')
        padded_mask = np.pad(blurred_mask, ((max(0, -sq_y1), max(0, sq_y2 - h)), (max(0, -sq_x1), max(0, sq_x2 - w))), mode='constant', constant_values=0)
        
        # Adjust indices for padded arrays
        p_y1, p_x1 = max(0, sq_y1), max(0, sq_x1)
        p_y2, p_x2 = p_y1 + side_len, p_x1 + side_len
        
        cropped_img = padded_img[p_y1:p_y2, p_x1:p_x2, :]
        cropped_mask = padded_mask[p_y1:p_y2, p_x1:p_x2]
        
        # 4. Resize to target resolutions (Upscale)
        img_for_resize = torch.from_numpy(cropped_img).unsqueeze(0).permute(0, 3, 1, 2) # [1, C, H, W]
        mask_for_resize = torch.from_numpy(cropped_mask).unsqueeze(0).unsqueeze(0)      # [1, 1, H, W]
        
        resized_img_chw = torch.nn.functional.interpolate(img_for_resize, size=(target_size, target_size), mode='bicubic', align_corners=False)
        resized_mask_chw = torch.nn.functional.interpolate(mask_for_resize, size=(target_size, target_size), mode='bilinear', align_corners=False)
        
        resized_img_bhwc = resized_img_chw.permute(0, 2, 3, 1) # [1, H, W, C]
        
        # 5. VAE Encode
        # vae.encode returns a latent tensor, which usually must be wrapped in dict
        latent_tensor = vae.encode(resized_img_bhwc[:,:,:,:3])
        latent = {"samples": latent_tensor}
        
        # 6. Set Noise Mask on Latent
        # KSampler expects noise_mask to match latent spatial dimensions [B, H, W]
        latent_mask = torch.nn.functional.interpolate(resized_mask_chw, size=(target_size // 8, target_size // 8), mode='nearest').squeeze(1)
        latent["noise_mask"] = latent_mask
        
        # 6.5 Evaluate Guide Prompts
        # Create empty conditionings first
        from nodes import CLIPTextEncode
        
        # Default empty conditionings
        empty_cond = CLIPTextEncode().encode(clip, "")[0]
        final_positive = empty_cond
        final_negative = empty_cond
        
        custom_parts = []
        if preset_prompt and preset_prompt != "None":
            custom_parts.append(preset_prompt)
        if guide_prompt and guide_prompt.strip():
            custom_parts.append(guide_prompt.strip())
            
        if custom_parts:
            combined_text = ", ".join(custom_parts)
            print(f"DEBUG: Encoding Detailer Custom Prompt: '{combined_text}'")
            final_positive = CLIPTextEncode().encode(clip, combined_text)[0]
        
        # 7. KSampler Inpaint
        # common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise)
        sampled_latent = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, final_positive, final_negative, latent, denoise=denoise)[0]
        
        # 8. VAE Decode
        decoded_img_bhwc = vae.decode(sampled_latent["samples"]) # [1, H, W, 3]
        decoded_img_chw = decoded_img_bhwc.permute(0, 3, 1, 2)
        
        # 9. Downscale and Stitch
        downscaled_img_chw = torch.nn.functional.interpolate(decoded_img_chw, size=(side_len, side_len), mode='area')
        downscaled_img_bhwc = downscaled_img_chw.permute(0, 2, 3, 1)
        
        # Paste back to padded base, then crop to original bounds
        result_padded = padded_img.copy()
        
        # Alpha blending on the padded regions
        alpha_mask = padded_mask[p_y1:p_y2, p_x1:p_x2, np.newaxis]
        result_padded[p_y1:p_y2, p_x1:p_x2, :] = (downscaled_img_bhwc[0].cpu().numpy() * alpha_mask) + (result_padded[p_y1:p_y2, p_x1:p_x2, :] * (1 - alpha_mask))
        
        # Unpad
        unpadded_result = result_padded[max(0, -sq_y1):max(0, -sq_y1)+h, max(0, -sq_x1):max(0, -sq_x1)+w, :]
        
        final_image = torch.from_numpy(unpadded_result).unsqueeze(0).float()
        
        print(f"✅ MidnightDetailer: Crop and Stitch complete with Denoise {denoise}.")
        return (final_image,)

NODE_CLASS_MAPPINGS = {
    "SAM2LoaderNode": SAM2LoaderNode,
    "MidnightDetailerNode": MidnightDetailerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM2LoaderNode": "SAM2 Loader (ML)",
    "MidnightDetailerNode": "Midnight Detailer (ML)",
}
