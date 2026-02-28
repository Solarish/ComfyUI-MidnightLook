import torch
import numpy as np
import comfy.utils
import nodes
import comfy_extras.nodes_upscale_model

class SampleUpscalerProviderNode:
    """
    Acts as a provider holding the necessary components and configuration 
    for the iterative upscaling loop.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("UPSCALER_PROVIDER",)
    RETURN_NAMES = ("provider",)
    FUNCTION = "process"
    CATEGORY = "MidnightLook/Upscale"

    def process(self, model, vae, upscale_model, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise):
        provider_dict = {
            "model": model,
            "vae": vae,
            "upscale_model": upscale_model,
            "positive": positive,
            "negative": negative,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise
        }
        return (provider_dict,)


class IterativeUpscaleNode:
    """
    Handles the Progressive Upscaling loop.
    Decodes latent, upscales via model, scales to iteration target size, 
    encodes to latent, refines via KSampler, and repeats.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "provider": ("UPSCALER_PROVIDER",),
                "scale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05}),
                "iterations": ("INT", {"default": 3, "min": 1, "max": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "process"
    CATEGORY = "MidnightLook/Upscale"

    def process(self, latent, provider, scale_factor, iterations):
        current_latent = latent
        
        # Calculate scaling multiplier per iteration
        step_scale = scale_factor ** (1.0 / iterations)
        
        vae = provider["vae"]
        upscale_model = provider["upscale_model"]
        
        # We use comfy_extras upscaler node internally
        upscale_model_node = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel()
        
        for i in range(iterations):
            print(f"🔄 Iterative Upscale: Step {i+1}/{iterations} (Scale: {step_scale:.3f}x)")
            
            # 1. Decode current latent
            img_tensor = vae.decode(current_latent["samples"]) # [B, H, W, C]
            b, h, w, c = img_tensor.shape
            
            # Target size for this step
            target_w = int(w * step_scale)
            target_h = int(h * step_scale)
            
            # 2. Model Upscale (usually 4x)
            # upscale_model_node expects (upscale_model, image)
            upscaled_img = upscale_model_node.upscale(upscale_model, img_tensor)[0] # [B, H', W', C]
            
            # 3. Resize to exact target size
            upscaled_chw = upscaled_img.permute(0, 3, 1, 2)
            resized_chw = torch.nn.functional.interpolate(upscaled_chw, size=(target_h, target_w), mode='bicubic', align_corners=False)
            resized_bhwc = resized_chw.permute(0, 2, 3, 1) # [B, H_T, W_T, C]
            
            # 4. Encode to Latent
            new_latent_tensor = vae.encode(resized_bhwc[:,:,:,:3]) # [B, C, H, W]
            new_latent = {"samples": new_latent_tensor}
            
            # 5. Refine (KSampler)
            sampled_latent = nodes.common_ksampler(
                model=provider["model"],
                seed=provider["seed"] + i, # vary seed slightly per step
                steps=provider["steps"],
                cfg=provider["cfg"],
                sampler_name=provider["sampler_name"],
                scheduler=provider["scheduler"],
                positive=provider["positive"],
                negative=provider["negative"],
                latent=new_latent,
                denoise=provider["denoise"]
            )[0]
            
            current_latent = sampled_latent
            final_img = resized_bhwc
        
        # Final Decode for the IMAGE output
        final_img = vae.decode(current_latent["samples"])
        
        print(f"✅ Iterative Upscale: Completed {iterations} iterations.")
        return (final_img, current_latent)

NODE_CLASS_MAPPINGS = {
    "SampleUpscalerProviderNode": SampleUpscalerProviderNode,
    "IterativeUpscaleNode": IterativeUpscaleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SampleUpscalerProviderNode": "Upscaler Provider (ML)",
    "IterativeUpscaleNode": "Iterative Upscale (ML)",
}
