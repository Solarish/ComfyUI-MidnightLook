import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image

# Dummy image
img = Image.new('RGB', (512, 512), color = 'red')
prompt = "face."

dino_path = r"C:\Dev\Docker\models\grounding-dino\grounding-dino-base"

print(f"Loading {dino_path}")
processor = AutoProcessor.from_pretrained(dino_path, local_files_only=True)
model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_path, local_files_only=True)

inputs = processor(images=img, text=prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

print("Raw Outputs:")
print(outputs.logits.shape)
print(outputs.logits.max())

try:
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[img.size[::-1]] # (height, width)
    )[0]
except TypeError:
    results = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=[img.size[::-1]]
    )[0]

boxes = results["boxes"]
scores = results["scores"]

print(f"Post Process Found {len(scores)} objects")
if len(scores) > 0:
    print(f"Max score: {scores.max().item():.4f}")
    
bt = 0.1
valid_indices = scores > bt
boxes = boxes[valid_indices]
scores = scores[valid_indices]

print(f"Objects remaining after threshold {bt}: {len(scores)}")
