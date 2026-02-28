import torch
import numpy as np
from PIL import Image

def get_comfy_image():
    img = Image.open("/app/ComfyUI/input/ComfyUI_temp_gbyoz_00002_.png").convert("RGB")
    r, g, b = img.split()
    return Image.merge("RGB", (b, g, r))

dino_path = "/app/ComfyUI/models/grounding-dino"
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

print(f"Loading from {dino_path}")
try:
    processor = AutoProcessor.from_pretrained(dino_path, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_path, local_files_only=True)
    print("Model loaded.")
except Exception as e:
    print(f"Error loading: {e}")
    exit(1)
    
img = get_comfy_image()

prompts = ["face."]
for prompt in prompts:
    inputs = processor(images=img, text=prompt, return_tensors="pt")
    print(f"Input IDs: {inputs.input_ids}")
    try:
        tokens = processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        print(f"Tokens: {tokens}")
    except Exception as e:
        print(f"Tokens error: {e}")
        
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"Prompt: {prompt} -> Max Logit: {outputs.logits.sigmoid().max().item():.4f}")

# How does post_process handle this?
print("--- Post Process Default ---")
try:
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[img.size[::-1]]
    )[0]
except TypeError:
    results = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=[img.size[::-1]]
    )[0]

print(f"Default objects found: {len(results['scores'])}")

print("--- Post Process with Threshold 0.0 ---")
try:
    results2 = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.0,
        text_threshold=0.0,
        target_sizes=[img.size[::-1]]
    )[0]
except TypeError:
    # try setting class attributes if it uses them?
    if hasattr(processor, "box_threshold"):
        processor.box_threshold = 0.0
        processor.text_threshold = 0.0
    results2 = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=[img.size[::-1]]
    )[0]

print(f"Threshold 0.0 objects found: {len(results2['scores'])}")
