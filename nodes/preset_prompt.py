import os
import json
import random

def load_presets():
    """Load presets from the JSON file. Defaults to an empty list if file not found."""
    # preset.json path: <root>/.agent/skills/comfyui/Preset-Prompt/preset.json
    preset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        ".agent", "skills", "comfyui", "Preset-Prompt", "preset.json"
    )
    if not os.path.exists(preset_path):
        return []
    try:
        with open(preset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading preset.json: {e}")
        return []

class MidnightLook_PresetPrompt:
    """
    A prompt builder node driven by an external JSON file (preset.json).
    Provides categorized inputs with presets for easier prompt construction.
    The JSON is read inside INPUT_TYPES to ensure it updates when ComfyUI reloads the node.
    """

    @classmethod
    def INPUT_TYPES(s):
        presets = load_presets()
        
        required_inputs = {
            "trigger_word": ("STRING", {"multiline": False, "default": ""}), 
            "randomize_presets": ("BOOLEAN", {"default": False}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }
        
        # Dynamically build inputs from JSON
        for preset_category in presets:
            cat_id = preset_category.get("id")
            if not cat_id:
                continue
            options = ["None"] + preset_category.get("options", [])
            
            # The combo list for preset selection
            required_inputs[f"{cat_id}_preset"] = (options, {"default": "None"})
            # The string input for custom text override/addition
            required_inputs[f"{cat_id}_custom"] = ("STRING", {"multiline": False, "default": ""})
            
        return {
            "required": required_inputs,
            "optional": {
                "description_input": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_prompt",)
    FUNCTION = "process"
    CATEGORY = "MidnightLook/Z-Image"

    def process(self, trigger_word, randomize_presets, seed, description_input="", **kwargs):
        rng = random.Random(seed)
        presets = load_presets()
        
        def get_preset_val(chosen, options_list):
            if randomize_presets:
                valid_options = [x for x in options_list if x not in ("None", "")]
                return rng.choice(valid_options) if valid_options else "None"
            return chosen

        parts = []

        # 1. Trigger Word
        if trigger_word.strip():
            parts.append(trigger_word.strip())

        # 2. Description Input (e.g. from Qwen or text prompt node)
        if description_input and description_input.strip():
            parts.append(description_input.strip())

        # 3. Dynamic Categories (Process in order of JSON)
        for preset_category in presets:
            cat_id = preset_category.get("id")
            if not cat_id:
                continue
                
            preset_key = f"{cat_id}_preset"
            custom_key = f"{cat_id}_custom"
            
            preset_val = kwargs.get(preset_key, "None")
            custom_val = kwargs.get(custom_key, "")
            
            options_list = preset_category.get("options", [])
            
            # Get actual preset value (considering randomizer)
            actual_preset_val = get_preset_val(preset_val, options_list)
            
            # Resolve custom override vs preset
            resolved_val = self._resolve(actual_preset_val, custom_val)
            if resolved_val:
                parts.append(resolved_val)

        # Combine all parts, removing any trailing commas
        final_prompt = ", ".join([p.strip().strip(",") for p in parts if p.strip()])
        
        return (final_prompt,)

    def _resolve(self, preset, custom):
        """Helper to combine preset and custom text. Prioritize custom text if provided."""
        if custom and custom.strip():
            return custom.strip()
        if preset and preset not in ("None", ""):
            return preset
        return ""
