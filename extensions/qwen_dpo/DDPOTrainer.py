import torch
import torch.nn.functional as F
import copy
import os
import gc
from jobs.process import BaseProcess
from diffusers import (
    FlowMatchEulerDiscreteScheduler, 
    Transformer2DModel, 
    SD3Transformer2DModel, 
    FluxTransformer2DModel
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import model_info

HF_TOKEN=""

class DDPOTrainer(BaseProcess):
    def __init__(self, process_id, job, config):
        super().__init__(process_id, job, config)
        self.device = torch.device(self.config.get('device', 'cuda'))
        self.dtype = torch.bfloat16 if self.config.get('dtype') == 'bf16' else torch.float16
        self.model_type = "generic" # Will be auto-detected

    def detect_and_load_model(self):
        model_id = self.config['model']['name_or_path']
        print(f"--- UNIVERSAL LOADER: Inspecting {model_id} ---")

        # 1. Inspect HuggingFace Config to find Architecture
        try:
            info = model_info(model_id)
            config_file = [f for f in info.siblings if f.rfilename.endswith("config.json")]
            # Heuristics based on model ID if config check is ambiguous
            if "stable-diffusion-3" in model_id.lower():
                print(f"Detected Architecture: Stable Diffusion 3 (MMDiT)")
                self.model_type = "sd3"
                return SD3Transformer2DModel.from_pretrained(model_id, subfolder="transformer", token=HF_TOKEN, torch_dtype=self.dtype)
            
            elif "flux" in model_id.lower():
                print(f"Detected Architecture: FLUX.1")
                self.model_type = "flux"
                return FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", token=HF_TOKEN, torch_dtype=self.dtype)
            
            elif "pixart" in model_id.lower() or "dit" in model_id.lower():
                print(f"Detected Architecture: Standard DiT (PixArt/DiT)")
                self.model_type = "dit"
                return Transformer2DModel.from_pretrained(model_id, subfolder="transformer", token=HF_TOKEN, torch_dtype=self.dtype)
            
            else:
                print(f"Architecture Unclear. Defaulting to Generic Transformer2DModel.")
                self.model_type = "dit"
                return Transformer2DModel.from_pretrained(model_id, subfolder="transformer", token=HF_TOKEN, torch_dtype=self.dtype)

        except Exception as e:
            print(f"Auto-detection failed: {e}")
            print("Fallback: Trying Generic Load...")
            self.model_type = "dit"
            return Transformer2DModel.from_pretrained(model_id, subfolder="transformer", token=HF_TOKEN, torch_dtype=self.dtype)

    def load_model(self):
        # 1. Auto-Load Correct Class
        self.model = self.detect_and_load_model()
        self.model.to(self.device)
        self.model.enable_gradient_checkpointing()

        # 2. Universal LoRA Injection
        # Targets the most common attention projection names across all architectures
        print(f"Injecting Universal LoRA...")
        target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",   # SD3 / DiT Standard
            "q_proj", "k_proj", "v_proj", "o_proj", # LLaMA / Qwen Style
            "add_k_proj", "add_v_proj"            # ID embeddings
        ]
        
        lora_config = LoraConfig(
            r=32, 
            lora_alpha=32,
            target_modules=target_modules, 
            lora_dropout=0.0,
            bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.model.train()

    def universal_forward(self, model, latents, t, prompt_embeds, pooled_embeds):
        """
        Adapts inputs to whatever weird format the specific model wants.
        """
        
        # --- STABLE DIFFUSION 3 ---
        if self.model_type == "sd3":
            return model(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False
            )[0]

        # --- FLUX ---
        elif self.model_type == "flux":
            # Flux usually requires 'guidance' (vector embedding)
            # If dataset doesn't provide it, we fake a standard guidance value (e.g. 3.5)
            bsz = latents.shape[0]
            guidance = torch.tensor([3.5] * bsz, device=self.device, dtype=self.dtype)
            
            return model(
                hidden_states=latents,
                timestep=t / 1000, # Flux often expects 0-1 range
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                guidance=guidance,
                return_dict=False
            )[0]

        # --- STANDARD DiT (PixArt, Qwen-Image, DiT-XL) ---
        else:
            # Most generic DiTs just take hidden_states, timestep, and encoder_hidden_states
            kwargs = {
                "hidden_states": latents,
                "timestep": t,
                "encoder_hidden_states": prompt_embeds,
                "return_dict": False
            }
            # Some DiTs (like Qwen) crash if class_labels isn't passed
            # We pass dummy zeros just in case the config requires it
            if hasattr(model.config, "num_embeds_ada_norm"):
                 bsz = latents.shape[0]
                 kwargs["class_labels"] = torch.zeros(bsz, device=self.device, dtype=torch.long)
            
            return model(**kwargs)[0]

    def run(self):
        self.load_model()
        
        from .DDPODataset import DDPODataset
        dataset = DDPODataset(self.config['dataset'])
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config['dataset'].get('batch_size', 1), shuffle=True
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        print(f"Starting Training ({self.model_type.upper()} Mode)...")
        beta = self.config.get('dpo_beta', 1000)
        global_step = 0

        max_steps = self.config.get('max_train_steps', None) 
        if max_steps:
            print(f"DEBUG: Training will stop after {max_steps} steps.")

        for epoch in range(self.config.get('num_epochs', 10)):
            for batch in dataloader:
                # Inputs
                latents_w = batch['w_latents'].to(self.device, dtype=self.dtype)
                latents_l = batch['l_latents'].to(self.device, dtype=self.dtype)
                p_embeds = batch['prompt_embeds'].to(self.device, dtype=self.dtype)
                pool_embeds = batch['pooled_embeds'].to(self.device, dtype=self.dtype)
                
                bsz = latents_w.shape[0]
                noise = torch.randn_like(latents_w)
                u = torch.rand((bsz,), device=self.device, dtype=self.dtype)
                timesteps = u * 1000 
                
                # Add Noise (Simple Flow Match Interpolation)
                u_view = u.view(bsz, 1, 1, 1)
                noisy_w = (1.0 - u_view) * latents_w + u_view * noise
                noisy_l = (1.0 - u_view) * latents_l + u_view * noise
                
                # --- UNIVERSAL FORWARD PASS ---
                pred_w = self.universal_forward(self.model, noisy_w, timesteps, p_embeds, pool_embeds)
                pred_l = self.universal_forward(self.model, noisy_l, timesteps, p_embeds, pool_embeds)
                
                # Ref Step
                with self.model.disable_adapter():
                    with torch.no_grad():
                        ref_pred_w = self.universal_forward(self.model, noisy_w, timesteps, p_embeds, pool_embeds)
                        ref_pred_l = self.universal_forward(self.model, noisy_l, timesteps, p_embeds, pool_embeds)

                # Loss
                target_w = noise - latents_w
                target_l = noise - latents_l

                loss_w = F.mse_loss(pred_w.float(), target_w.float(), reduction="none").mean([1, 2, 3])
                loss_l = F.mse_loss(pred_l.float(), target_l.float(), reduction="none").mean([1, 2, 3])
                ref_loss_w = F.mse_loss(ref_pred_w.float(), target_w.float(), reduction="none").mean([1, 2, 3])
                ref_loss_l = F.mse_loss(ref_pred_l.float(), target_l.float(), reduction="none").mean([1, 2, 3])

                logits = (ref_loss_w - loss_w) - (ref_loss_l - loss_l)
                loss = -torch.nn.functional.logsigmoid(beta * logits).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                if global_step % 10 == 0:
                    print(f"Step {global_step} Loss: {loss.item():.4f}")

                if global_step >= max_steps:
                    print("DEBUG: Reached max test steps. Stopping.")
                    self.model.save_pretrained(self.config['save_root'])
                    return
        
        self.model.save_pretrained(self.config['save_root'])
