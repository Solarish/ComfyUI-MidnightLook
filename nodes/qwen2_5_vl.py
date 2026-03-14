import os
import torch
import folder_paths
import numpy as np
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import uuid
from pathlib import Path

def get_vlm_dir():
    base = os.path.join(folder_paths.models_dir, "VLM")
    if os.path.exists(folder_paths.models_dir):
        for d in os.listdir(folder_paths.models_dir):
            if d.lower() == "vlm" and os.path.isdir(os.path.join(folder_paths.models_dir, d)):
                return os.path.join(folder_paths.models_dir, d)
    return base

def find_model_folders(base_path, max_depth=3):
    model_folders = []
    if not os.path.exists(base_path):
        return model_folders
        
    for root, dirs, files in os.walk(base_path):
        rel_root = os.path.relpath(root, base_path)
        depth = 0 if rel_root == "." else len(rel_root.split(os.sep))
        
        if "config.json" in files:
            if rel_root != ".":
                model_folders.append(rel_root)
            
        if depth >= max_depth:
            dirs[:] = [] # Stop going deeper
            continue
            
    return sorted(model_folders)

# Helper functions for temp files (same as original but cleaner if possible)
def temp_image(image, seed):
    unique_id = uuid.uuid4().hex
    image_path = (
        Path(folder_paths.temp_directory) / f"temp_image_{seed}_{unique_id}.png"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )
    img.save(os.path.join(image_path))
    return f"file://{image_path.as_posix()}"

def temp_batch_image(image, num_counts, seed):
    image_batch_path = Path(folder_paths.temp_directory) / "Multiple"
    image_batch_path.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for Nth_count in range(num_counts):
        img = Image.fromarray(
            np.clip(255.0 * image[Nth_count].cpu().numpy().squeeze(), 0, 255).astype(
                np.uint8
            )
        )
        unique_id = uuid.uuid4().hex
        image_path = image_batch_path / f"temp_image_{seed}_{Nth_count}_{unique_id}.png"
        img.save(os.path.join(image_path))
        image_paths.append(f"file://{image_path.resolve().as_posix()}")
    return image_paths

def temp_video(video, seed):
    unique_id = uuid.uuid4().hex
    video_path = (
        Path(folder_paths.temp_directory) / f"temp_video_{seed}_{unique_id}.mp4"
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    # create parent if not exists
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if video object has save_to method (it should be VideoInput object)
    if hasattr(video, "save_to"):
        video.save_to(
            os.path.join(video_path),
            format="mp4",
            codec="h264",
        )
    else:
        # Fallback or error if video input is not what we expect
        # For now assume it works as in original node
        pass

    return f"{video_path.as_posix()}"


class MidnightQwen25Load:
    @classmethod
    def INPUT_TYPES(s):
        # Scan VLM folder recursively
        vlm_dir = get_vlm_dir()
        if not os.path.exists(vlm_dir):
            os.makedirs(vlm_dir, exist_ok=True)
            
        # Get model folders recursively
        models = find_model_folders(vlm_dir)
        if not models:
            models = ["No models found in models/VLM"]
            
        return {
            "required": {
                "model": (models,),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("QWEN2_5_VL_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MidnightLook/Qwen"

    def load_model(self, model, device, precision):
        if model == "No models found in models/VLM":
            raise ValueError("No Qwen2.5-VL models found in ComfyUI/models/VLM/. Please download one.")
            
        vlm_dir = get_vlm_dir()
        model_path = os.path.join(vlm_dir, model)
        
        # Auto-drill down: If config.json is not in the path, look one level deeper
        if not os.path.exists(os.path.join(model_path, "config.json")):
            print(f"config.json not found in {model_path}. Searching subfolders...")
            try:
                for d in os.listdir(model_path):
                    sub_path = os.path.join(model_path, d)
                    if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "config.json")):
                        print(f"Found model files in: {sub_path}")
                        model_path = sub_path
                        break
            except Exception as e:
                print(f"Error during auto-drill down: {e}")

        print(f"Loading Qwen2.5-VL model from: {model_path}")
        
        # Determine dtype
        torch_dtype = torch.float16
        if precision == "bf16":
            torch_dtype = torch.bfloat16
        elif precision == "fp32":
            torch_dtype = torch.float32

        # Load model
        try:
            model_obj = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device,
                attn_implementation="flash_attention_2" if device == "cuda" else "eager",
            )
        except Exception as e:
            print(f"Error loading model with flash_attention_2, trying default: {e}")
            model_obj = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device,
            )

        # Load processor
        processor = AutoProcessor.from_pretrained(model_path)
            
        return ({"model": model_obj, "processor": processor, "path": model_path},)


class MidnightQwen25Run:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN2_5_VL_MODEL",),
                "system_text": ("STRING", {
                    "default": "You are an observant AI assistant specialized in describing human physical traits.\nFocus primarily on:\n1. Body build (slim, muscular, chubby, average, etc.)\n2. Height estimation (tall, short, average)\n3. Skin tone and complexion\n4. Hair color, style, and length\n5. Facial features (face shape, cheekbones, eyes, nose, lips)\nDo not describe clothes or background unless necessary for context.", 
                    "multiline": True
                }),
                "text": ("STRING", {"default": "Describe the physical appearance of the person in the image in detail.", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "min_pixels": ("INT", {"default": 256, "min": 64, "max": 1280}),
                "max_pixels": ("INT", {"default": 1280, "min": 64, "max": 2048}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "MidnightLook/Qwen"

    def run(self, model, system_text, text, max_new_tokens, min_pixels, max_pixels, seed, image=None, video=None):
        qwen_model = model["model"]
        processor = model["processor"]
        
        # Prepare content list
        content = []
        
        # Helper pixel values
        min_pixels = min_pixels * 28 * 28
        max_pixels = max_pixels * 28 * 28
        
        if image is not None:
            num_counts = image.shape[0]
            if num_counts == 1:
                uri = temp_image(image, seed)
                content.append({
                    "type": "image",
                    "image": uri,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                })
            else:
                image_paths = temp_batch_image(image, num_counts, seed)
                for path in image_paths:
                    content.append({
                        "type": "image",
                        "image": path,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    })

        if video is not None:
             # Assuming video input format from ComfyUI (which might be a wrapper)
             # The original node used `video` directly in `temp_video`.
             # We should wrap this in try-except or check type if possible.
             try:
                uri = temp_video(video, seed)
                content.append({
                    "type": "video",
                    "video": uri,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                })
             except Exception as e:
                 print(f"Error processing video input: {e}")

        # Add text prompt
        if text:
            content.append({"type": "text", "text": text})

        # Construct messages
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": content},
        ]

        # Prepare for inference
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        
        # --- FIX FOR TRANSFORMERS 5.2.0 / FPS ERROR ---
        # The issue is that `video_kwargs` might contain `fps=[[]]` or similar empty structures 
        # when no video is present or if qwen_vl_utils fails to extract it properly.
        # Transformers expect `fps` to be valid numbers or None.
        
        if "fps" in video_kwargs:
            fps_val = video_kwargs.get("fps")
            # If fps is an empty list [], it causes TypeError in transformers 5.2.0
            if isinstance(fps_val, list) and len(fps_val) == 0:
                del video_kwargs["fps"]
            elif fps_val is None:
                pass # None is usually fine, but if it causes issues we can remove it too


        # Validating inputs before call
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        
        inputs = inputs.to(qwen_model.device)
        
        # Generate
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        # Cleanup temp files if needed (optional, temp folder gets cleaned up eventually or we can leave it)
        
        return (output_text[0],)


NODE_CLASS_MAPPINGS = {
    "MidnightQwen25Load": MidnightQwen25Load,
    "MidnightQwen25Run": MidnightQwen25Run,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MidnightQwen25Load": "MidnightLook Qwen2.5 Load",
    "MidnightQwen25Run": "MidnightLook Qwen2.5 Run",
}
