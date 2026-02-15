import torch


class MidnightLook_LatentSizePresets:
    """Creates an empty latent tensor based on predefined common image sizes."""

    SIZE_MAPPINGS = {
        "1024x1024 (1:1 Square)": (1024, 1024),
        "768x1024 (3:4 Portrait)": (768, 1024),
        "896x1152 (7:9 Portrait)": (896, 1152),
        "1024x1344 (16:21 Portrait)": (1024, 1344),
        # Landscape
        "1024x768 (4:3 Landscape)": (1024, 768),
        "1152x896 (9:7 Landscape)": (1152, 896),
        "1344x1024 (21:16 Landscape)": (1344, 1024),
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "size_preset": (list(s.SIZE_MAPPINGS.keys()),),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "MidnightLook/Latent"

    def generate_latent(self, size_preset, batch_size):
        width, height = self.SIZE_MAPPINGS[size_preset]

        # Latent space dimensions are 1/8th of pixel space
        latent_width = width // 8
        latent_height = height // 8

        latent = torch.zeros([batch_size, 4, latent_height, latent_width])
        print(
            f"✅ MidnightLook (Latent Size): "
            f"Created empty latent {width}x{height} (batch: {batch_size})"
        )
        return ({"samples": latent},)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "MidnightLook_LatentSizePresets": MidnightLook_LatentSizePresets,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MidnightLook_LatentSizePresets": "Latent Size Presets (ML)",
}
