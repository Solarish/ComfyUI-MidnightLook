import os
import sys
import importlib.util

file_path = r"c:\Dev\Docker\custom_nodes\ComfyUI-MidnightLook\nodes\preset_prompt.py"
spec = importlib.util.spec_from_file_location("preset_prompt", file_path)
preset_prompt = importlib.util.module_from_spec(spec)
sys.modules["preset_prompt"] = preset_prompt
spec.loader.exec_module(preset_prompt)

inputs = preset_prompt.MidnightLook_PresetPrompt.INPUT_TYPES()
print("INPUT_TYPES:")
import pprint
pprint.pprint(inputs)

