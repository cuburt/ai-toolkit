import os
import json
import torch
import gc
import glob
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline, FluxPipeline, PixArtSigmaPipeline
from diffusers.utils import load_image
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
HF_TOKEN = os.environ.get("HF_TOKEN", "")


# ==========================================
# PHASE 1: GENERATION
# ==========================================
def load_local_llm(model_id):
    print(f"Loading local LLM: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    return tokenizer, model


def enhance_prompt_local(base_prompt, target_style, tokenizer, model):
    system_prompt = (
        f"You are an expert prompt engineer for the {target_style} image model. "
        "Expand the following simple prompt into a highly detailed, visually descriptive "
        "prompt optimized for this architecture. Reply ONLY with the new prompt, no chat."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": base_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
    generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def run_generation(args):
    out_dir = Path(args.output_dir)
    accepted_dir, rejected_dir, source_dir = out_dir / "accepted", out_dir / "rejected", out_dir / "source"
    for d in [accepted_dir, rejected_dir, source_dir]: d.mkdir(parents=True, exist_ok=True)

    items = []
    print(f"Reading {args.jsonl_file} in {args.mode.upper()} mode...")
    with open(args.jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line.strip())
            if args.prompt_key not in item: continue
            if args.mode == "edit" and args.image_path_key not in item: continue
            items.append(item)
            if 0 < args.max_samples <= len(items): break

    if not items:
        print("No valid prompts/images found! Exiting.");
        return

    tokenizer, llm = load_local_llm(args.llm_model)
    enhanced_prompts = []
    for item in items:
        enhanced_prompts.append(enhance_prompt_local(item[args.prompt_key], args.target_style, tokenizer, llm))

    print("Purging LLM from VRAM...")
    del llm, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nLoading Generation Pipeline: {args.image_model}...")
    pipe = DiffusionPipeline.from_pretrained(args.image_model, torch_dtype=torch.bfloat16).to("cuda")
    pipe.enable_model_cpu_offload()

    metadata_path = out_dir / "metadata.jsonl"
    with open(metadata_path, 'a', encoding='utf-8') as metadata_file:
        for idx, (item, enhanced_prompt) in enumerate(zip(items, enhanced_prompts)):
            base_prompt = item[args.prompt_key]
            acc_path = accepted_dir / f"{idx:04d}_accepted.png"
            rej_path = rejected_dir / f"{idx:04d}_rejected.png"
            entry = {"base_prompt": base_prompt, "enhanced_prompt": enhanced_prompt,
                     "accepted_image": str(acc_path.name), "rejected_image": str(rej_path.name)}

            if args.mode == "edit":
                try:
                    init_image = load_image(item[args.image_path_key]).convert("RGB").resize((1024, 1024))
                except Exception as e:
                    print(f"Skipping pair {idx}: {e}"); continue

                rejected_img = pipe(prompt=base_prompt, image=init_image, num_inference_steps=args.steps, strength=0.9,
                                    generator=torch.Generator("cuda").manual_seed(args.seed + idx)).images[0]
                accepted_img = \
                pipe(prompt=enhanced_prompt, image=init_image, num_inference_steps=args.steps, strength=0.7,
                     generator=torch.Generator("cuda").manual_seed(args.seed + idx)).images[0]

                src_path = source_dir / f"{idx:04d}_source.png"
                init_image.save(src_path)
                entry["source_image"] = str(src_path.name)
            else:
                rejected_img = pipe(prompt=base_prompt, num_inference_steps=args.steps,
                                    generator=torch.Generator("cuda").manual_seed(args.seed + idx)).images[0]
                accepted_img = pipe(prompt=enhanced_prompt, num_inference_steps=args.steps,
                                    generator=torch.Generator("cuda").manual_seed(args.seed + idx)).images[0]

            accepted_img.save(acc_path)
            rejected_img.save(rej_path)
            metadata_file.write(json.dumps(entry) + "\n")
            print(f"Generated pair {idx + 1}/{len(items)}")

    print("Purging Generation Pipeline from VRAM...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


# ==========================================
# PHASE 2: PREPROCESSING (LATENTS)
# ==========================================
def load_encoding_pipeline(model_id, device):
    dtype = torch.bfloat16
    if "stable-diffusion-3" in model_id.lower():
        return StableDiffusion3Pipeline.from_pretrained(model_id, token=HF_TOKEN, torch_dtype=dtype).to(device), "sd3"
    elif "flux" in model_id.lower():
        return FluxPipeline.from_pretrained(model_id, token=HF_TOKEN, torch_dtype=dtype).to(device), "flux"
    elif "pixart" in model_id.lower():
        return PixArtSigmaPipeline.from_pretrained(model_id, token=HF_TOKEN, torch_dtype=dtype).to(device), "pixart"
    else:
        raise ValueError("Unknown architecture")


def run_preprocessing(args):
    print(f"\nLoading Encoding Pipeline: {args.image_model}...")
    out_dir = Path(args.output_dir)
    latents_dir = out_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    pipe, arch = load_encoding_pipeline(args.image_model, device)
    pipe.set_progress_bar_config(disable=True)
    vae = pipe.vae

    acc_files = glob.glob(os.path.join(out_dir / "accepted", "*_accepted.*"))

    # Load metadata mapping for prompts
    metadata = {}
    meta_path = out_dir / "metadata.jsonl"
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                metadata[data["accepted_image"]] = data["enhanced_prompt"]

    for acc_path in acc_files:
        basename = os.path.basename(acc_path).replace("_accepted", "").split('.')[0]
        ext = os.path.splitext(acc_path)[1]

        rej_path = os.path.join(out_dir / "rejected", f"{basename}_rejected{ext}")
        src_path = os.path.join(out_dir / "source", f"{basename}_source{ext}")

        has_source = os.path.exists(src_path)
        prompt = metadata.get(os.path.basename(acc_path), "default prompt")

        if not os.path.exists(rej_path): continue

        try:
            with torch.no_grad():
                if arch == "flux":
                    prompt_embeds, pooled_embeds = pipe.encode_prompt(prompt=prompt, prompt_2=prompt)[0:2]
                else:
                    prompt_embeds, pooled_embeds = pipe.encode_prompt(prompt=prompt, prompt_2=prompt, prompt_3=prompt)[
                        0], pipe.encode_prompt(prompt=prompt, prompt_2=prompt, prompt_3=prompt)[2]

                def encode_img(path):
                    img = Image.open(path).convert("RGB").resize((1024, 1024))
                    img_t = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(device,
                                                                                         dtype=vae.dtype) / 127.5 - 1.0
                    latents = vae.encode(img_t).latent_dist.sample()
                    return (
                                       latents - vae.config.shift_factor) * vae.config.scaling_factor if arch == "flux" else latents * vae.config.scaling_factor

                data_dict = {
                    "w_latents": encode_img(acc_path).detach().cpu(),
                    "l_latents": encode_img(rej_path).detach().cpu(),
                    "prompt_embeds": prompt_embeds.detach().cpu(),
                    "pooled_embeds": pooled_embeds.detach().cpu(),
                    "caption": prompt
                }

                if has_source: data_dict["s_latents"] = encode_img(src_path).detach().cpu()

            torch.save(data_dict, os.path.join(latents_dir, f"{basename}.pt"))
            print(f"Encoded and saved {basename}.pt")
        except Exception as e:
            print(f"Error encoding {basename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, choices=["generate", "preprocess", "all"], default="all")
    parser.add_argument("--mode", type=str, choices=["standard", "edit"], default="standard")
    parser.add_argument("--jsonl_file", type=str, required=True)
    parser.add_argument("--prompt_key", type=str, default="image_prompt")
    parser.add_argument("--image_path_key", type=str, default="image_path")
    parser.add_argument("--output_dir", type=str, default="./dpo_dataset")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--image_model", type=str, required=True)
    parser.add_argument("--target_style", type=str, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    if args.step in ["generate", "all"]: run_generation(args)
    if args.step in ["preprocess", "all"]: run_preprocessing(args)