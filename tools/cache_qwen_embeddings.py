import torch
import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Setup
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_DIR = "path/to/my_dpo_dataset"  # Should contain 'prompts/' folder with .txt files

print("Loading Qwen VLM...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).cuda()
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Ensure output dir exists
os.makedirs(os.path.join(DATA_DIR, "embeddings"), exist_ok=True)
prompt_files = [f for f in os.listdir(os.path.join(DATA_DIR, "prompts")) if f.endswith('.txt')]

print(f"Processing {len(prompt_files)} prompts...")

for p_file in prompt_files:
    # Read Prompt
    with open(os.path.join(DATA_DIR, "prompts", p_file), 'r') as f:
        text = f.read().strip()

    # Prepare Inputs
    # Note: Qwen-Image usually expects specific formatting (e.g. <|image_pad|>)
    # Ensure this matches your inference setup exactly.
    inputs = processor(text=[text], images=None, return_tensors="pt").to("cuda")

    with torch.no_grad():
        # Forward pass through the Transformer
        outputs = model.model(**inputs)

        # Extract embeddings needed for DiT
        # These keys depend on specific Qwen-Image implementation details
        # Typically: last_hidden_state and a pooled vector (like EOS token)
        data = {
            "prompt_embeds": outputs.last_hidden_state.cpu().float(),
            "pooled_embeds": outputs.pooler_output.cpu().float()
            # If pooler_output is None, grab last token: outputs.last_hidden_state[:, -1]
        }

    # Save
    save_name = os.path.splitext(p_file)[0] + ".pt"
    torch.save(data, os.path.join(DATA_DIR, "embeddings", save_name))

print("Done!")