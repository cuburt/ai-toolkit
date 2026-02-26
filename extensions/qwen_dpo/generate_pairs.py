import os
import json
import torch
import gc
import argparse
from pathlib import Path
from diffusers import DiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_local_llm(model_id):
    """Loads a lightweight local LLM to save VRAM."""
    print(f"Loading local LLM: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
        "prompt optimized for this architecture. Reply ONLY with the new prompt, no chat. "
        "Do not include quotes or conversational filler."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_prompt}
    ]

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

    # --- LOAD JSONL ---
    base_prompts = []
    print(f"Reading prompts from {args.jsonl_file} using key '{args.prompt_key}'...")
    with open(args.jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if args.prompt_key in item:
                    base_prompts.append(item[args.prompt_key])

                    # Stop reading if we hit the user's limit
                    if args.max_samples > 0 and len(base_prompts) >= args.max_samples:
                        print(f"Reached max_samples limit ({args.max_samples}). Stopping read.")
                        break
                else:
                    print(f"Warning: Key '{args.prompt_key}' not found on line {line_num}. Skipping.")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON on line {line_num}. Skipping.")

    if not base_prompts:
        print("No valid prompts found! Exiting.")
        return

    print(f"Successfully loaded {len(base_prompts)} prompts to process.")

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
    pipe.enable_model_cpu_offload()

    metadata_path = out_dir / "metadata.jsonl"
    with open(metadata_path, 'a', encoding='utf-8') as metadata_file:
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
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs locally from JSONL.")
    parser.add_argument("--jsonl_file", type=str, required=True, help="Path to input.jsonl")
    parser.add_argument("--prompt_key", type=str, default="image_prompt", help="Key containing the prompt in JSONL")
    parser.add_argument("--output_dir", type=str, default="./dpo_dataset")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace LLM ID")
    parser.add_argument("--image_model", type=str, required=True, help="HuggingFace image model ID")
    parser.add_argument("--target_style", type=str, required=True, help="Instructions for the LLM")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0, help="Max number of prompts to process (0 = all)")

    args = parser.parse_args()
    main(args)