import torch
import os
import glob
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusion3Pipeline, 
    FluxPipeline, 
    PixArtSigmaPipeline
)

# --- CONFIGURATION ---
HF_TOKEN=""
SOURCE_FOLDER = "/workspace/BATCH_GENERATION/outputs"   # Folder with *-positive.jpg, *-negative.jpg, *.txt
OUTPUT_FOLDER = "/workspace/BATCH_GENERATION/sd3/latents"   # Where .pt files go

# CHANGE THIS to your target model:
# Options:
# "stabilityai/stable-diffusion-3.5-medium"
# "black-forest-labs/FLUX.1-dev"
# "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium" 
# ---------------------

def load_pipeline_components(model_id, device):
    """Auto-detects architecture and loads correct VAE & Text Encoders"""
    dtype = torch.bfloat16
    print(f"--- Auto-Detecting Architecture for {model_id} ---")

    if "stable-diffusion-3" in model_id.lower():
        print(">> Architecture: SD3 (MMDiT)")
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, token=HF_TOKEN, torch_dtype=dtype).to(device)
        return pipe, "sd3"

    elif "flux" in model_id.lower():
        print(">> Architecture: FLUX.1")
        pipe = FluxPipeline.from_pretrained(model_id, token=HF_TOKEN, torch_dtype=dtype).to(device)
        return pipe, "flux"

    elif "pixart" in model_id.lower():
        print(">> Architecture: PixArt-Sigma")
        pipe = PixArtSigmaPipeline.from_pretrained(model_id, token=HF_TOKEN, torch_dtype=dtype).to(device)
        return pipe, "pixart"

    else:
        raise ValueError("Unknown model architecture. Please use SD3, Flux, or PixArt.")

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    device = "cuda"
    
    # 1. Load Universal Pipeline
    pipe, arch = load_pipeline_components(MODEL_ID, device)
    pipe.set_progress_bar_config(disable=True)
    vae = pipe.vae
    
    # 2. Find Negative Samples (The Anchor)
    # We look for files ending in "-negative.png/jpg"
    neg_files = glob.glob(os.path.join(SOURCE_FOLDER, "*-negative.*"))
    print(f"Found {len(neg_files)} negative samples.")

    for neg_path in neg_files:
        basename = os.path.basename(neg_path).replace("-negative", "").split('.')[0]
        ext = os.path.splitext(neg_path)[1]
        
        # Construct paths for Positive pair and Prompt
        pos_path = os.path.join(SOURCE_FOLDER, f"{basename}-positive{ext}")
        
        # Try finding text file (basename.txt OR basename-prompt.txt)
        txt_path = os.path.join(SOURCE_FOLDER, f"{basename}.txt") 
        if not os.path.exists(txt_path):
             txt_path = os.path.join(SOURCE_FOLDER, f"{basename}-prompt.txt")

        if not os.path.exists(pos_path) or not os.path.exists(txt_path):
            print(f"Skipping {basename}: Missing pair or text.")
            continue

        try:
            with torch.no_grad():
                # --- A. ENCODE TEXT (Architecture Specific) ---
                with open(txt_path, 'r') as f:
                    prompt = f.read().strip()

                if arch == "sd3":
                    # SD3: 3 Text Encoders -> returns (prompt_embeds, _, pooled_embeds, _)
                    out = pipe.encode_prompt(prompt=prompt, prompt_2=prompt, prompt_3=prompt)
                    prompt_embeds = out[0]
                    pooled_embeds = out[2]

                elif arch == "flux":
                    # Flux: 2 Text Encoders -> returns (prompt_embeds, pooled_embeds, ids)
                    out = pipe.encode_prompt(prompt=prompt, prompt_2=prompt)
                    prompt_embeds = out[0]
                    pooled_embeds = out[1]

                elif arch == "pixart":
                    # PixArt: T5 Encoder -> returns (prompt_embeds, attention_mask)
                    prompt_embeds, _ = pipe.encode_prompt(prompt=prompt)
                    pooled_embeds = torch.zeros(1, 1) # Dummy for consistency

                # --- B. ENCODE IMAGES (Winner & Loser) ---
                def encode_img(path):
                    img = Image.open(path).convert("RGB").resize((1024, 1024))
                    img_t = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(device, dtype=vae.dtype)
                    img_t = img_t / 127.5 - 1.0 # Normalize -1 to 1
                    
                    latents = vae.encode(img_t).latent_dist.sample()
                    
                    # Flux requires specific latent shifting
                    if arch == "flux":
                        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                    else:
                        latents = latents * vae.config.scaling_factor
                    return latents

                latents_w = encode_img(pos_path).detach().cpu()
                latents_l = encode_img(neg_path).detach().cpu()

            # --- C. SAVE ---
            save_path = os.path.join(OUTPUT_FOLDER, f"{basename}.pt")
            torch.save({
                "w_latents": latents_w,
                "l_latents": latents_l,
                "prompt_embeds": prompt_embeds.detach().cpu(),
                "pooled_embeds": pooled_embeds.detach().cpu(),
                "caption": prompt
            }, save_path)
            
            print(f"Processed {basename} ({arch})")
            
        except Exception as e:
            print(f"Error processing {basename}: {e}")

if __name__ == "__main__":
    main()