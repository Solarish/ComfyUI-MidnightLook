class MidnightLook_ZImagePrompt:
    """
    A prompt builder node optimized for Z-Image Turbo.
    Provides categorized inputs with presets for easier prompt construction.
    """

    # --- PRESETS ---
    SUBJECT_PRESETS = [
        "None",
        "",
        "A 20-year-old Asian woman",
        "A 25-year-old Caucasian woman", 
        "A 30-year-old Black woman",
        "A 25-year-old Asian man",
        "A 30-year-old Caucasian man",
        "A cute cat",
        "A golden retriever dog",
    ]
    
    OUTFIT_PRESETS = [
        "None",
        "",
        "Casual T-shirt and jeans",
        "Elegant evening dress",
        "Business suit",
        "Cyberpunk techwear",
        "Traditional kimono",
        "School uniform",
        "Sportswear",
        "Bikini",
    ]
    
    POSE_PRESETS = [
        "None",
        "",
        "Standing gracefully",
        "Sitting relaxed",
        "Walking towards camera",
        "Looking at viewer",
        "Action jumping pose",
        "Close-up headshot",
        "Side profile",
    ]
    
    BACKGROUND_PRESETS = [
        "None",
        "",
        "City street at night",
        "Sunny beach",
        "Forest nature",
        "Modern living room",
        "Studio plain background",
        "Cyberpunk city",
        "Coffee shop",
    ]

    LIGHTING_PRESETS = {
        "None": "",
        "": "",
        "Natural (Golden Hour)": "golden hour sunlight, warm atmosphere",
        "Natural (Overcast)": "soft diffused light, overcast sky",
        "Studio (Rembrandt)": "Rembrandt lighting, dramatic shadows",
        "Studio (Softbox)": "soft studio lighting, even illumination",
        "Cinematic (Cyberpunk)": "neon lights, cyberpunk atmosphere, blue and pink hues",
        "Night (Moonlight)": "moonlight, dark atmosphere, cold tones",
    }
    
    STYLE_PRESETS = {
        "None": "",
        "": "",
        "Film (Leica M6/Portra 400)": "Shot on Leica M6, Kodak Portra 400 film grain, 35mm lens, photorealistic, highly detailed",
        "Vintage (35mm/Raw)": "Captured with a vintage 35mm lens, visible film grain, slight lens softness, chromatic aberration, 16mm glow, raw photo",
        "Portrait (Analog/Gritty)": "Gritty analog film photograph, natural film grain, shallow depth of field, warm and muted color grading",
        "Studio (Canon 5D/8K)": "Shot on Canon 5D with 85mm lens, shallow depth of field, sharp focus, 8k resolution, masterpiece", 
        "Fashion (High-End)": "High-fashion photography, professional studio lighting, 100mm lens, f/2.8, extreme detail, 8k resolution",
        "Action (Sports/Motion)": "Action photography, sports photography, fast shutter speed, central subject in sharp focus, motion blur on background",
        "Social (iPhone Selfie)": "Shot on iPhone, top-down selfie angle, realistic iPhone grain, candid snapshot",
        "Social (Flash/Party)": "Under a sudden iPhone flash, flash-lit snapshot, casual youth photography",
        "Cinematic (Movie Still)": "Cinematic movie still, dramatic volumetric lighting, anamorphic lens flare, movie poster style",
        "Art (Sci-Fi Concept)": "High-octane Sci-Fi concept art, painterly but detailed, electric atmosphere",
        "Art (Minimalist Arch)": "Architectural minimalist photography, high-key, monochromatic color palette, symmetrical composition",
        "Art (Oil Paint)": "Heavy impasto texture, mimicking thick oil paint strokes applied with a palette knife, surreal and symbolic",
    }
    
    TEXT_RENDERING_PRESETS = [
        "None",
        "",
        "holding a sign saying 'Hello'",
        "wearing a t-shirt with text 'Love'",
        "neon sign reading 'Open'",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trigger_word": ("STRING", {"multiline": False, "default": ""}), 
                "randomize_presets": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                
                # Combos offer list but allow custom text if widgets support it (ComfyUI combo usually restricts to list).
                # To support custom text alongside presets in standard Comfy nodes, 
                # we usually provide a Combo AND a separate String input, or rely on custom frontend.
                # Here, we will provide the COMBO for quick select, and a separate "custom_..." string input for override/addition?
                # The user requested "Combos + Custom STRING input check". 
                # Standard ComfyUI Combos are strict lists. 
                # Strategy: We provide the List. If user wants custom, they pick "None" (Reference) or specific item AND type in a separate field?
                # OR input is just STRING? 
                
                # Actually, many custom nodes use a trick or just simple STRING for everything.
                # But user asked for "PRESETS".
                # Let's use STRING everywhere for maximum flexibility, BUT populate the validation list? No.
                # Let's use COMBOs for the presets, and STRINGs for custom overrides/additions.
                # Wait, simpler: Just STRING inputs but with `dynamicPrompts` style? No.
                
                # Let's try to implement "Combo + Custom" Logic:
                # We will have pairs: [Style Select] + [Style Custom]. 
                # If Style Select != None, we use it. If Style Custom != "", we append it? 
                # Or if Style Select == "Custom", use Style Custom?
                
                # User said: "list ให้เลือก หรือจะ custom เองก็ได้"
                # Let's go with:
                # 1. Preset Dropdown (Default: None)
                # 2. Custom Text (Default: "")
                # Final = Preset + " " + Custom
                
                "subject_preset": (s.SUBJECT_PRESETS, {"default": "None"}),
                "subject_custom": ("STRING", {"multiline": False, "default": ""}),

                "outfit_preset": (s.OUTFIT_PRESETS, {"default": "None"}),
                "outfit_custom": ("STRING", {"multiline": False, "default": ""}),

                "pose_preset": (s.POSE_PRESETS, {"default": "None"}),
                "pose_custom": ("STRING", {"multiline": False, "default": ""}),
                
                "background_preset": (s.BACKGROUND_PRESETS, {"default": "None"}),
                "background_custom": ("STRING", {"multiline": False, "default": ""}),

                "lighting_preset": (list(s.LIGHTING_PRESETS.keys()), {"default": "None"}),
                "lighting_custom": ("STRING", {"multiline": False, "default": ""}),

                "style_preset": (list(s.STYLE_PRESETS.keys()), {"default": "None"}),
                "style_custom": ("STRING", {"multiline": False, "default": ""}),

                "text_rendering_preset": (s.TEXT_RENDERING_PRESETS, {"default": "None"}),
                "text_rendering_custom": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "description_input": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_prompt",)
    FUNCTION = "process"
    CATEGORY = "MidnightLook/Z-Image"

    def process(self, trigger_word, randomize_presets, seed,
                subject_preset, subject_custom,
                outfit_preset, outfit_custom,
                pose_preset, pose_custom,
                background_preset, background_custom,
                lighting_preset, lighting_custom,
                style_preset, style_custom,
                text_rendering_preset, text_rendering_custom,
                description_input=""):
        
        import random
        rng = random.Random(seed)

        def get_preset_val(chosen, preset_collection):
            if randomize_presets:
                if isinstance(preset_collection, dict):
                    options = [k for k in preset_collection.keys() if k not in ("None", "")]
                    return rng.choice(options) if options else "None"
                else:
                    options = [x for x in preset_collection if x not in ("None", "")]
                    return rng.choice(options) if options else "None"
            return chosen

        parts = []

        # 1. Trigger Word
        if trigger_word.strip():
            parts.append(trigger_word.strip())

        # 1.5 Description Input (from Qwen or other)
        if description_input and description_input.strip():
            parts.append(description_input.strip())

        # 2. Subject
        subject_p = get_preset_val(subject_preset, self.SUBJECT_PRESETS)
        subject = self._resolve(subject_p, subject_custom)
        if subject: parts.append(subject)

        # 3. Outfit
        outfit_p = get_preset_val(outfit_preset, self.OUTFIT_PRESETS)
        outfit = self._resolve(outfit_p, outfit_custom)
        if outfit: parts.append(outfit)

        # 4. Pose
        pose_p = get_preset_val(pose_preset, self.POSE_PRESETS)
        pose = self._resolve(pose_p, pose_custom)
        if pose: parts.append(pose)

        # 5. Background
        bg_p = get_preset_val(background_preset, self.BACKGROUND_PRESETS)
        bg = self._resolve(bg_p, background_custom)
        if bg: parts.append(bg)

        # 6. Lighting (Map key to value)
        light_p = get_preset_val(lighting_preset, self.LIGHTING_PRESETS)
        light_val = self.LIGHTING_PRESETS.get(light_p, "")
        if lighting_custom.strip():
            light_val = lighting_custom.strip()
        if light_val.strip():
            parts.append(light_val.strip(","))

        # 7. Style (Map key to value)
        style_p = get_preset_val(style_preset, self.STYLE_PRESETS)
        style_val = self.STYLE_PRESETS.get(style_p, "")
        if style_custom.strip():
            style_val = style_custom.strip()
        if style_val.strip():
            parts.append(style_val.strip(","))

        # 8. Text Rendering
        text_p = get_preset_val(text_rendering_preset, self.TEXT_RENDERING_PRESETS)
        text_val = self._resolve(text_p, text_rendering_custom)
        if text_val: parts.append(text_val)

        # Combine
        final_prompt = ", ".join([p.strip().strip(",") for p in parts if p.strip()])
        
        return (final_prompt,)

    def _resolve(self, preset, custom):
        """Helper to combine preset and custom text. Prioritize custom text if provided."""
        if custom and custom.strip():
            return custom.strip()
        if preset and preset not in ("None", ""):
            return preset
        return ""

NODE_CLASS_MAPPINGS = {
    "MidnightLook_ZImagePrompt": MidnightLook_ZImagePrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MidnightLook_ZImagePrompt": "Z-Image Prompt Prep (ML)",
}
