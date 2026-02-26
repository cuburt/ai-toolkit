import copy
import os
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from jobs.process import BaseProcess
from diffusers import FlowMatchEulerDiscreteScheduler, Transformer2DModel, SD3Transformer2DModel, FluxTransformer2DModel
from peft import LoraConfig, get_peft_model

# --- MONKEY PATCH FOR PYTORCH 2.4 ---
_original_sdpa = F.scaled_dot_product_attention


def _patched_sdpa(*args, **kwargs):
    kwargs.pop('enable_gqa', None)
    return _original_sdpa(*args, **kwargs)


F.scaled_dot_product_attention = _patched_sdpa

try:
    from diffusers import Flux2Transformer2DModel
except ImportError:
    Flux2Transformer2DModel = FluxTransformer2DModel

HF_TOKEN = os.environ.get("HF_TOKEN", "")


# ==========================================
# DATASET CLASS
# ==========================================
class DDPODataset(Dataset):
    def __init__(self, config):
        self.root = config['root_path']
        self.files = [f for f in os.listdir(self.root) if f.endswith('.pt')]
        if not self.files: print(f"WARNING: No .pt files found in {self.root}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        data = torch.load(path, map_location='cpu', weights_only=True)

        out = {
            "w_latents": data['w_latents'].squeeze(0),
            "l_latents": data['l_latents'].squeeze(0),
            "prompt_embeds": data['prompt_embeds'].squeeze(0),
            "pooled_embeds": data['pooled_embeds'].squeeze(0)
        }
        # Safely pass source latents if they exist (Edit mode)
        if 's_latents' in data:
            out["s_latents"] = data['s_latents'].squeeze(0)

        return out


# ==========================================
# TRAINER CLASS
# ==========================================
class Projector(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dtype):
        super().__init__()
        self.net = torch.nn.Linear(in_dim, out_dim, bias=False).to(dtype=dtype)
        torch.nn.init.normal_(self.net.weight, std=0.02)

    def forward(self, x): return self.net(x)


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
        try:
            if "flux.2-klein" in model_id.lower() or "flux-2-klein" in model_id.lower():
                self.model_type = "flux2_klein"
                return Flux2Transformer2DModel.from_pretrained(model_id, token=HF_TOKEN, subfolder="transformer",
                                                               torch_dtype=self.dtype)
            elif "flux" in model_id.lower():
                self.model_type = "flux"
                return FluxTransformer2DModel.from_pretrained(model_id, token=HF_TOKEN, subfolder="transformer",
                                                              torch_dtype=self.dtype)
            else:
                self.model_type = "dit"
                return Transformer2DModel.from_pretrained(model_id, token=HF_TOKEN, subfolder="transformer",
                                                          torch_dtype=self.dtype)
        except Exception:
            self.model_type = "dit"
            return Transformer2DModel.from_pretrained(model_id, token=HF_TOKEN, subfolder="transformer",
                                                      torch_dtype=self.dtype)

    def load_model(self):
        self.model = self.detect_and_load_model().to(self.device)
        self.model.enable_gradient_checkpointing()
        target_modules = ["to_q", "to_k", "to_v", "to_out.0", "q_proj", "k_proj", "v_proj", "o_proj", "add_k_proj",
                          "add_v_proj"]
        self.model = get_peft_model(self.model,
                                    LoraConfig(r=32, lora_alpha=32, target_modules=target_modules, bias="none"))
        self.model.print_trainable_parameters()
        self.model.train()

    def check_and_create_adapters(self, batch):
        data_dim = batch['prompt_embeds'].shape[-1]
        model_dim = getattr(self.model.config, "joint_attention_dim", 7680) if self.model_type in ["flux",
                                                                                                   "flux2_klein"] else getattr(
            self.model.config, "cross_attention_dim", data_dim)
        if data_dim != model_dim and self.prompt_adapter is None:
            self.prompt_adapter = Projector(data_dim, model_dim, self.dtype).to(self.device)

        if 'pooled_embeds' in batch:
            pooled_data_dim = batch['pooled_embeds'].shape[-1]
            pooled_model_dim = 768 if self.model_type in ["flux", "flux2_klein"] else pooled_data_dim
            if pooled_data_dim != pooled_model_dim and pooled_model_dim != 1 and self.pooled_adapter is None:
                self.pooled_adapter = Projector(pooled_data_dim, pooled_model_dim, self.dtype).to(self.device)

    def universal_forward(self, model, latents, t, prompt_embeds, pooled_embeds):
        if self.prompt_adapter: prompt_embeds = self.prompt_adapter(prompt_embeds)
        if self.pooled_adapter and pooled_embeds is not None and pooled_embeds.shape[-1] > 1:
            pooled_embeds = self.pooled_adapter(pooled_embeds)

        if self.model_type in ["flux", "flux2_klein"]:
            bsz, c, h, w = latents.shape
            latents_packed = latents.view(bsz, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(bsz,
                                                                                                          (h // 2) * (
                                                                                                                      w // 2),
                                                                                                          c * 4)

            txt_ids = torch.zeros((bsz, prompt_embeds.shape[1], 4), device=self.device, dtype=self.dtype)
            img_ids = torch.zeros((h // 2, w // 2, 4), device=self.device, dtype=self.dtype)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=self.device, dtype=self.dtype)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=self.device, dtype=self.dtype)[None, :]
            img_ids = img_ids.view((h // 2) * (w // 2), 4).unsqueeze(0).repeat(bsz, 1, 1)

            guidance = torch.tensor([3.5] * bsz, device=self.device, dtype=self.dtype)

            if self.model_type == "flux2_klein":
                pred = model(hidden_states=latents_packed, timestep=t / 1000, encoder_hidden_states=prompt_embeds,
                             guidance=guidance, txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
            else:
                pred = model(hidden_states=latents_packed, timestep=t / 1000, encoder_hidden_states=prompt_embeds,
                             pooled_projections=pooled_embeds, guidance=guidance, txt_ids=txt_ids, img_ids=img_ids,
                             return_dict=False)[0]

            c_out = 16
            return pred.reshape(bsz, h // 2, w // 2, c_out, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(bsz, c_out, h, w)
        else:
            return model(hidden_states=latents, timestep=t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

    def run(self):
        self.load_model()
        dataset = DDPODataset(self.config['dataset'])
        dataloader = DataLoader(dataset, batch_size=self.config['dataset'].get('batch_size', 1), shuffle=True)

        sample_batch = next(iter(dataloader))
        self.check_and_create_adapters(sample_batch)

        params_to_optimize = list(self.model.parameters())
        if self.prompt_adapter: params_to_optimize += list(self.prompt_adapter.parameters())
        if self.pooled_adapter: params_to_optimize += list(self.pooled_adapter.parameters())
        optimizer = torch.optim.AdamW(params_to_optimize, lr=float(self.config.get('optimizer', {}).get('lr', 1e-5)))

        beta = self.config.get('dpo_beta', 200)
        max_steps = self.config.get('max_train_steps', None)
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

                # UNIVERSAL TENSOR PACKING
                if 's_latents' in batch:
                    # Edit mode: Map Source Latents to Noisy Latents (32 channels)
                    latents_s = batch['s_latents'].to(self.device, dtype=self.dtype)
                    input_w = torch.cat([noisy_w, latents_s], dim=1)
                    input_l = torch.cat([noisy_l, latents_s], dim=1)
                else:
                    # Standard mode: Proceed directly (16 channels)
                    input_w = noisy_w
                    input_l = noisy_l

                pred_w = self.universal_forward(self.model, input_w, timesteps, p_embeds, pool_embeds)
                pred_l = self.universal_forward(self.model, input_l, timesteps, p_embeds, pool_embeds)

                with self.model.disable_adapter():
                    with torch.no_grad():
                        ref_pred_w = self.universal_forward(self.model, input_w, timesteps, p_embeds, pool_embeds)
                        ref_pred_l = self.universal_forward(self.model, input_l, timesteps, p_embeds, pool_embeds)

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
                    self.model.save_pretrained(self.config['save_root'])
                    return

        self.model.save_pretrained(self.config['save_root'])