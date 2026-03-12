import os
import hashlib
import requests
import folder_paths
import comfy.utils
import comfy.sd
from tqdm import tqdm

class MidnightLook_URLLoRALoader:
    """
    Downloads a LoRA from a URL with caching and robustness.
    Categories: MidnightLook/Loaders
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "url": ("STRING", {"multiline": False}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_url_lora"
    CATEGORY = "MidnightLook/Loaders"

    def load_url_lora(self, model, clip, url, strength_model, strength_clip):
        if not url or not url.strip():
            return (model, clip)

        # 1. Determine download directory
        # We try to put it in a subfolder of the first lora path
        lora_paths = folder_paths.get_folder_paths("loras")
        if not lora_paths:
            # Fallback to models/loras
            download_dir = os.path.join(folder_paths.models_dir, "loras", "url_downloads")
        else:
            download_dir = os.path.join(lora_paths[0], "url_downloads")

        if not os.path.exists(download_dir):
            os.makedirs(download_dir, exist_ok=True)

        # 2. Generate unique filename based on URL hash
        url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
        
        # Try to infer extension
        ext = ".safetensors"
        if url.lower().endswith(".pt") or url.lower().endswith(".ckpt"):
             ext = ".pt"
        elif url.lower().endswith(".safetensors"):
             ext = ".safetensors"
        
        filename = f"{url_hash}{ext}"
        full_path = os.path.join(download_dir, filename)

        # 3. Check if cached
        if not os.path.exists(full_path):
            print(f"📡 MidnightLook: Downloading LoRA from {url}...")
            try:
                # Use a temp file for atomic download
                temp_path = full_path + ".downloading"
                
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024 # 1MB
                
                progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="MidnightLook Download")
                
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))
                
                progress_bar.close()
                
                # Atomic rename
                os.rename(temp_path, full_path)
                print(f"✅ MidnightLook: LoRA downloaded successfully to {full_path}")
                
            except Exception as e:
                print(f"❌ MidnightLook Error: Failed to download LoRA from {url}")
                print(f"   Reason: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return (model, clip)

        # 4. Load LoRA
        print(f"📦 MidnightLook: Loading LoRA {filename}...")
        try:
            lora = comfy.utils.load_torch_file(full_path, safe_load=True)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
            return (model_lora, clip_lora)
        except Exception as e:
            print(f"❌ MidnightLook Error: Failed to load LoRA from {full_path}")
            print(f"   Reason: {e}")
            return (model, clip)

NODE_CLASS_MAPPINGS = {
    "MidnightLook_URLLoRALoader": MidnightLook_URLLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MidnightLook_URLLoRALoader": "Load LoRA from URL (ML)",
}
