import os
import yaml
import json
import torch
import gc
import argparse
from pathlib import Path
from diffusers import DiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_local_llm(model_id):
    """Loads a lightweight local LLM in 8-bit to save VRAM."""
    print(f"Loading local LLM: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Loading in bfloat16 to match your diffusion dtype.
    # Use load_in_8bit=True if you are super tight on VRAM.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    return tokenizer, model


def enhance_prompt_local(base_prompt, target_style, tokenizer, model):
    """Uses the local LLM to rewrite the prompt."""
    system_prompt = (
        f"You are an expert prompt engineer for the {target_style} image model. "
        "Expand the following simple prompt into a highly detailed, visually descriptive "
        "prompt optimized for this architecture. Reply ONLY with the new prompt, no chat."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_prompt}
    ]

    # Format for the specific instruct model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True
    )

    # Strip the input context from the output
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


def main(args):
    # Set up directories
    out_dir = Path(args.output_dir)
    accepted_dir = out_dir / "accepted"
    rejected_dir = out_dir / "rejected"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    # Load base prompts
    with open(args.yaml_file, 'r') as f:
        base_prompts = yaml.safe_load(f).get('prompts', [])

    print(f"Loaded {len(base_prompts)} prompts from {args.yaml_file}.")

    # --- PHASE 1: TEXT GENERATION ---
    tokenizer, llm = load_local_llm(args.llm_model)

    enhanced_prompts = []
    print("\n--- Generating Enhanced Prompts ---")
    for idx, base_prompt in enumerate(base_prompts):
        enhanced = enhance_prompt_local(base_prompt, args.target_style, tokenizer, llm)
        enhanced_prompts.append(enhanced)
        print(f"[{idx + 1}/{len(base_prompts)}] Base: {base_prompt}")
        print(f"      Enhanced: {enhanced}\n")

    # CRITICAL VRAM PURGE: Delete the LLM before loading FLUX
    print("Purging LLM from VRAM to make room for the Image Model...")
    del llm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- PHASE 2: IMAGE GENERATION ---
    print(f"\nLoading Image Pipeline: {args.image_model}...")
    pipe = DiffusionPipeline.from_pretrained(
        args.image_model,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.enable_model_cpu_offload()  # Extra safety for VRAM

    metadata_path = out_dir / "metadata.jsonl"
    with open(metadata_path, 'a') as metadata_file:
        for idx, (base_prompt, enhanced_prompt) in enumerate(zip(base_prompts, enhanced_prompts)):
            print(f"\nGenerating Image Pair {idx + 1}/{len(base_prompts)}...")

            # Rejected Image (Base Prompt)
            gen_rej = torch.Generator("cuda").manual_seed(args.seed + idx)
            rejected_img = pipe(
                prompt=base_prompt,
                num_inference_steps=args.steps,
                generator=gen_rej
            ).images[0]

            # Accepted Image (Enhanced Prompt)
            gen_acc = torch.Generator("cuda").manual_seed(args.seed + idx)
            accepted_img = pipe(
                prompt=enhanced_prompt,
                num_inference_steps=args.steps,
                generator=gen_acc
            ).images[0]

            # Save Outputs
            accepted_path = accepted_dir / f"{idx:04d}_accepted.png"
            rejected_path = rejected_dir / f"{idx:04d}_rejected.png"
            accepted_img.save(accepted_path)
            rejected_img.save(rejected_path)

            # Log Metadata
            entry = {
                "base_prompt": base_prompt,
                "enhanced_prompt": enhanced_prompt,
                "accepted_image": str(accepted_path.name),
                "rejected_image": str(rejected_path.name)
            }
            metadata_file.write(json.dumps(entry) + "\n")

    print(f"\n✅ Local DPO Dataset generation complete! Saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs locally.")
    parser.add_argument("--yaml_file", type=str, required=True, help="Path to prompts.yaml")
    parser.add_argument("--output_dir", type=str, default="./dpo_dataset")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace LLM ID")
    parser.add_argument("--image_model", type=str, required=True, help="HuggingFace image model ID")
    parser.add_argument("--target_style", type=str, required=True, help="Instructions for the LLM")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)