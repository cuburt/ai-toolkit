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
import torch
import torch.nn.functional as F

# --- MONKEY PATCH: STRIP 'enable_gqa' FOR PYTORCH 2.4 ---
_original_sdpa = F.scaled_dot_product_attention


def _patched_sdpa(*args, **kwargs):
    kwargs.pop('enable_gqa', None)  # Quietly remove the future keyword
    return _original_sdpa(*args, **kwargs)


F.scaled_dot_product_attention = _patched_sdpa
# --------------------------------------------------------

# Safe import for FLUX.2 Klein, gracefully falling back to FLUX.1 if alias is missing
try:
    from diffusers import Flux2Transformer2DModel
except ImportError:
    Flux2Transformer2DModel = FluxTransformer2DModel

from peft import LoraConfig, get_peft_model
from huggingface_hub import model_info

HF_TOKEN = ""


class Projector(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dtype):
        super().__init__()
        self.net = torch.nn.Linear(in_dim, out_dim, bias=False).to(dtype=dtype)
        torch.nn.init.normal_(self.net.weight, std=0.02)

    def forward(self, x):
        return self.net(x)


class DDPOLoRATrainer(BaseProcess):
    def __init__(self, process_id, job, config):
        super().__init__(process_id, job, config)
        self.device = torch.device(self.config.get('device', 'cuda'))
        self.dtype = torch.bfloat16 if self.config.get('dtype') == 'bf16' else torch.float16
        self.model_type = "generic"

        self.prompt_adapter = None
        self.pooled_adapter = None

    def detect_and_load_model(self):
        model_id = self.config['model']['name_or_path']
        print(f"--- UNIVERSAL LOADER: Inspecting {model_id} ---")

        try:
            if "stable-diffusion-3" in model_id.lower():
                print(f"Detected Architecture: Stable Diffusion 3 (MMDiT)")
                self.model_type = "sd3"
                return SD3Transformer2DModel.from_pretrained(model_id, token=HF_TOKEN, subfolder="transformer",
                                                             low_cpu_mem_usage=False, torch_dtype=self.dtype)

            elif "flux.2-klein" in model_id.lower() or "flux-2-klein" in model_id.lower():
                print(f"Detected Architecture: FLUX.2 [klein]")
                self.model_type = "flux2_klein"
                return Flux2Transformer2DModel.from_pretrained(model_id, token=HF_TOKEN, subfolder="transformer",
                                                               low_cpu_mem_usage=False, torch_dtype=self.dtype)

            elif "flux" in model_id.lower():
                print(f"Detected Architecture: FLUX.1")
                self.model_type = "flux"
                return FluxTransformer2DModel.from_pretrained(model_id, token=HF_TOKEN, subfolder="transformer",
                                                              low_cpu_mem_usage=False, torch_dtype=self.dtype)

            elif "pixart" in model_id.lower() or "dit" in model_id.lower():
                print(f"Detected Architecture: Standard DiT")
                self.model_type = "dit"
                return Transformer2DModel.from_pretrained(model_id, token=HF_TOKEN, subfolder="transformer",
                                                          low_cpu_mem_usage=False, torch_dtype=self.dtype)

            else:
                print(f"Architecture Unclear. Defaulting to Generic DiT.")
                self.model_type = "dit"
                return Transformer2DModel.from_pretrained(model_id, token=HF_TOKEN, subfolder="transformer",
                                                          low_cpu_mem_usage=False, torch_dtype=self.dtype)

        except Exception as e:
            print(f"Fallback: Trying Generic Load... ({e})")
            self.model_type = "dit"
            return Transformer2DModel.from_pretrained(model_id, token=HF_TOKEN, subfolder="transformer",
                                                      low_cpu_mem_usage=False, torch_dtype=self.dtype)

    def load_model(self):
        self.model = self.detect_and_load_model()
        self.model.to(self.device)
        self.model.enable_gradient_checkpointing()

        print(f"Injecting Universal LoRA...")
        target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",
            "q_proj", "k_proj", "v_proj", "o_proj",
            "add_k_proj", "add_v_proj"
        ]

        lora_config = LoraConfig(
            r=32, lora_alpha=32, target_modules=target_modules,
            lora_dropout=0.0, bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.model.train()

    def check_and_create_adapters(self, batch):
        data_dim = batch['prompt_embeds'].shape[-1]

        # --- DYNAMIC DIMENSION DETECTION ---
        if self.model_type == "sd3":
            model_dim = 4096
        elif self.model_type in ["flux", "flux2_klein"]:
            # Let the model tell us what it wants (FLUX.1 = 4096, FLUX.2 Klein = 7680)
            model_dim = getattr(self.model.config, "joint_attention_dim", 7680)
        else:
            model_dim = getattr(self.model.config, "cross_attention_dim", data_dim)

        if data_dim != model_dim:
            if self.prompt_adapter is None:
                print(f"--- BRIDGE CREATED: Resizing Prompt Embeds {data_dim} -> {model_dim} ---")
                self.prompt_adapter = Projector(data_dim, model_dim, self.dtype).to(self.device)

        if 'pooled_embeds' in batch:
            pooled_data_dim = batch['pooled_embeds'].shape[-1]

            if self.model_type == "sd3":
                try:
                    pooled_model_dim = self.model.time_text_embed.text_embedder.linear_1.in_features
                except:
                    pooled_model_dim = 2048
            elif self.model_type in ["flux", "flux2_klein"]:
                pooled_model_dim = 768
            else:
                pooled_model_dim = pooled_data_dim

            # Only create a pooled adapter if dimensions mismatch AND the model actually uses pooled embeds
            if pooled_data_dim != pooled_model_dim and pooled_model_dim != 1:
                if self.pooled_adapter is None:
                    print(f"--- BRIDGE CREATED: Resizing Pooled Embeds {pooled_data_dim} -> {pooled_model_dim} ---")
                    self.pooled_adapter = Projector(pooled_data_dim, pooled_model_dim, self.dtype).to(self.device)

    def universal_forward(self, model, latents, t, prompt_embeds, pooled_embeds):

        # --- APPLY ADAPTERS ---
        if self.prompt_adapter:
            prompt_embeds = self.prompt_adapter(prompt_embeds)

        if self.pooled_adapter and pooled_embeds is not None and pooled_embeds.shape[-1] > 1:
            pooled_embeds = self.pooled_adapter(pooled_embeds)

        # --- MODEL FORWARD ---
        if self.model_type == "sd3":
            return model(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False
            )[0]

        elif self.model_type in ["flux", "flux2_klein"]:
            bsz, c, h, w = latents.shape

            # 1. Pack Latents (Flux expects a 1D sequence, not a 2D image)
            latents_packed = latents.view(bsz, c, h // 2, 2, w // 2, 2)
            latents_packed = latents_packed.permute(0, 2, 4, 1, 3, 5)
            latents_packed = latents_packed.reshape(bsz, (h // 2) * (w // 2), c * 4)

            # 2. Generate RoPE IDs (Flux requires these for spatial awareness)
            txt_ids = torch.zeros((bsz, prompt_embeds.shape[1], 4), device=self.device, dtype=self.dtype)

            img_ids = torch.zeros((h // 2, w // 2, 4), device=self.device, dtype=self.dtype)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=self.device, dtype=self.dtype)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=self.device, dtype=self.dtype)[None, :]
            img_ids = img_ids.view((h // 2) * (w // 2), 4).unsqueeze(0).repeat(bsz, 1, 1)

            guidance = torch.tensor([3.5] * bsz, device=self.device, dtype=self.dtype)

            # 3. Forward Pass
            if self.model_type == "flux2_klein":
                # FLUX.2 drops the pooled_projections parameter completely
                pred = model(
                    hidden_states=latents_packed,
                    timestep=t / 1000,
                    encoder_hidden_states=prompt_embeds,
                    guidance=guidance,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    return_dict=False
                )[0]
            else:
                pred = model(
                    hidden_states=latents_packed,
                    timestep=t / 1000,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    guidance=guidance,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    return_dict=False
                )[0]

            # 4. Unpack back to standard 2D so our MSE loss math works correctly
            pred = pred.reshape(bsz, h // 2, w // 2, c, 2, 2)
            pred = pred.permute(0, 3, 1, 4, 2, 5)
            pred = pred.reshape(bsz, c, h, w)

            return pred

        else:
            kwargs = {
                "hidden_states": latents, "timestep": t,
                "encoder_hidden_states": prompt_embeds, "return_dict": False
            }
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

        sample_batch = next(iter(dataloader))
        self.check_and_create_adapters(sample_batch)

        params_to_optimize = list(self.model.parameters())
        if self.prompt_adapter: params_to_optimize += list(self.prompt_adapter.parameters())
        if self.pooled_adapter: params_to_optimize += list(self.pooled_adapter.parameters())

        optimizer = torch.optim.AdamW(params_to_optimize, lr=float(self.config.get('optimizer', {}).get('lr', 1e-5)))

        print(f"Starting Training ({self.model_type.upper()} Mode)...")
        beta = self.config.get('dpo_beta', 200)  # Lowered default for stability

        max_steps = self.config.get('max_train_steps', None)
        if max_steps:
            print(f"DEBUG: Training will stop after {max_steps} steps.")

        global_step = 0

        for epoch in range(self.config.get('num_epochs', 10)):
            for batch in dataloader:
                latents_w = batch['w_latents'].to(self.device, dtype=self.dtype)
                latents_l = batch['l_latents'].to(self.device, dtype=self.dtype)
                p_embeds = batch['prompt_embeds'].to(self.device, dtype=self.dtype)
                pool_embeds = batch['pooled_embeds'].to(self.device, dtype=self.dtype)

                bsz = latents_w.shape[0]
                noise = torch.randn_like(latents_w)
                u = torch.rand((bsz,), device=self.device, dtype=self.dtype)
                timesteps = u * 1000

                u_view = u.view(bsz, 1, 1, 1)
                noisy_w = (1.0 - u_view) * latents_w + u_view * noise
                noisy_l = (1.0 - u_view) * latents_l + u_view * noise

                pred_w = self.universal_forward(self.model, noisy_w, timesteps, p_embeds, pool_embeds)
                pred_l = self.universal_forward(self.model, noisy_l, timesteps, p_embeds, pool_embeds)

                with self.model.disable_adapter():
                    with torch.no_grad():
                        ref_pred_w = self.universal_forward(self.model, noisy_w, timesteps, p_embeds, pool_embeds)
                        ref_pred_l = self.universal_forward(self.model, noisy_l, timesteps, p_embeds, pool_embeds)

                target_w = noise - latents_w
                target_l = noise - latents_l

                loss_w = F.mse_loss(pred_w.float(), target_w.float(), reduction="none").mean([1, 2, 3])
                loss_l = F.mse_loss(pred_l.float(), target_l.float(), reduction="none").mean([1, 2, 3])
                ref_loss_w = F.mse_loss(ref_pred_w.float(), target_w.float(), reduction="none").mean([1, 2, 3])
                ref_loss_l = F.mse_loss(ref_pred_l.float(), target_l.float(), reduction="none").mean([1, 2, 3])

                logits = (ref_loss_w - loss_w) - (ref_loss_l - loss_l)
                loss = -torch.nn.functional.logsigmoid(beta * logits).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                if global_step % 10 == 0:
                    print(f"Step {global_step} Loss: {loss.item():.4f}")

                if max_steps and global_step >= max_steps:
                    print(f"Reaching max_train_steps ({max_steps}). Saving and stopping.")
                    self.model.save_pretrained(self.config['save_root'])
                    return

        self.model.save_pretrained(self.config['save_root'])