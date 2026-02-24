import torch
import torch.nn.functional as F
import copy
import json
import os
import inspect
import gc
from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download
from jobs.process import BaseProcess
from diffusers import FlowMatchEulerDiscreteScheduler, Transformer2DModel
from safetensors.torch import load_file

class DDPOTrainer(BaseProcess):
    def __init__(self, process_id, job, config):
        super().__init__(process_id, job, config)
        self.device = torch.device(self.config.get('device', 'cuda'))
        self.dtype = torch.bfloat16 if self.config.get('dtype') == 'bf16' else torch.float16

    def load_model(self):
        model_id = self.config['model']['name_or_path']
        print(f"Loading Qwen-Image DiT: {model_id}")

        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
            use_dynamic_shifting=False
        )

        try:
            # 1. DOWNLOAD CONFIG & WEIGHTS
            print(f"DEBUG: Ensuring weights are present...")
            local_dir = snapshot_download(repo_id=model_id, allow_patterns=["transformer/*"])
            transformer_path = os.path.join(local_dir, "transformer")

            config_file = os.path.join(transformer_path, "config.json")
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file missing at {config_file}")

            with open(config_file, 'r') as f:
                raw_config = json.load(f)

            # 2. CONSTRUCT "NUCLEAR" CONFIG (Safe Defaults)
            final_config = {
                "sample_size": 128,           
                "patch_size": 2,              
                "in_channels": 4,             
                "num_layers": 28,             
                "attention_head_dim": 72,     
                "num_attention_heads": 16,    
                "out_channels": 4,            
                "cross_attention_dim": 3584,  
                "norm_type": "ada_norm_zero", 
                "num_embeds_ada_norm": 1000,  
                "dropout": 0.0,
                "activation_fn": "gelu-approximate",
                "use_linear_projection": True
            }

            # 3. MERGE REPO CONFIG
            valid_args = set(inspect.signature(Transformer2DModel.__init__).parameters.keys())
            for key, value in raw_config.items():
                if key in valid_args and key not in ["_class_name", "architectures", "_name_or_path"]:
                    final_config[key] = value

            # 4. CRITICAL OVERRIDES
            final_config["cross_attention_dim"] = 3584
            final_config["norm_type"] = "ada_norm_zero"
            if "sample_size" not in final_config: final_config["sample_size"] = 128

            # 5. INITIALIZE MODEL (Empty Shell on CPU)
            print("DEBUG: Initializing Model Shell (CPU)...")
            self.model = Transformer2DModel(**final_config)
            
            # 6. STREAMING WEIGHT LOADING (Low RAM Mode)
            print("DEBUG: Streaming weights (Iterative Loading)...")
            weight_files = [f for f in os.listdir(transformer_path) if f.endswith('.safetensors')]
            weight_files.sort()
            
            for i, w_file in enumerate(weight_files):
                w_path = os.path.join(transformer_path, w_file)
                print(f"  - Loading shard {i+1}/{len(weight_files)}: {w_file} ...", end="", flush=True)
                
                # Load SINGLE shard to RAM
                shard_dict = load_file(w_path)
                
                # Update Model (In-Place)
                mismatches = self.model.load_state_dict(shard_dict, strict=False)
                
                # DELETE IMMEDIATELY
                del shard_dict
                gc.collect()
                print(" Done. RAM Cleared.")
            
            print("SUCCESS: All shards loaded.")

            # 7. MOVE TO GPU (Clears System RAM, Fills VRAM)
            print(f"DEBUG: Moving Model to GPU ({self.device})...")
            self.model.to(self.device, dtype=self.dtype)
            torch.cuda.empty_cache()

            # 8. CREATE REFERENCE MODEL (On GPU)
            # We copy from GPU-to-GPU, avoiding CPU RAM entirely
            print("DEBUG: Creating Reference Model (VRAM Copy)...")
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.requires_grad_(False)
            self.ref_model.eval()

        except Exception as e:
            print(f"\nCRITICAL ERROR: Failed to load {model_id}")
            print(f"Error Details: {e}")
            raise e
            
        self.model.enable_gradient_checkpointing()

    def get_dit_prediction(self, model, latents, t, prompt_embeds, pooled_embeds):
        kwargs = {
            "hidden_states": latents,
            "timestep": t,
            "return_dict": False,
            "encoder_hidden_states": prompt_embeds
        }
        bsz = latents.shape[0]
        kwargs["class_labels"] = torch.zeros(bsz, device=self.device, dtype=torch.long)
        return model(**kwargs)[0]

    def run(self):
        self.load_model()
        
        from .DDPODataset import DDPODataset
        dataset = DDPODataset(self.config['dataset'])
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config['dataset'].get('batch_size', 1), 
            shuffle=True
        )
        
        optimizer = self.get_optimizer(self.model.parameters())
        self.model.train()
        
        print(f"Starting Training...")
        beta = self.config.get('dpo_beta', 1000)
        global_step = 0

        for epoch in range(self.config.get('num_epochs', 10)):
            for batch in dataloader:
                latents_w = batch['w_latents'].to(self.device, dtype=self.dtype)
                latents_l = batch['l_latents'].to(self.device, dtype=self.dtype)
                p_embeds = batch['prompt_embeds'].to(self.device, dtype=self.dtype)
                pool_embeds = batch['pooled_embeds'].to(self.device, dtype=self.dtype)
                
                bsz = latents_w.shape[0]
                
                noise = torch.randn(latents_w.shape, device=self.device, dtype=self.dtype)
                u = torch.rand((bsz,), device=self.device, dtype=self.dtype)
                u_view = u.view(bsz, 1, 1, 1)
                
                noisy_w = (1.0 - u_view) * latents_w + u_view * noise
                noisy_l = (1.0 - u_view) * latents_l + u_view * noise
                timesteps = (u * 1000).long().to(self.device)

                pred_w = self.get_dit_prediction(self.model, noisy_w, timesteps, p_embeds, pool_embeds)
                pred_l = self.get_dit_prediction(self.model, noisy_l, timesteps, p_embeds, pool_embeds)
                
                with torch.no_grad():
                    ref_pred_w = self.get_dit_prediction(self.ref_model, noisy_w, timesteps, p_embeds, pool_embeds)
                    ref_pred_l = self.get_dit_prediction(self.ref_model, noisy_l, timesteps, p_embeds, pool_embeds)

                target_w = noise - latents_w
                target_l = noise - latents_l

                loss_w = F.mse_loss(pred_w, target_w, reduction="none").mean([1, 2, 3])
                loss_l = F.mse_loss(pred_l, target_l, reduction="none").mean([1, 2, 3])
                ref_loss_w = F.mse_loss(ref_pred_w, target_w, reduction="none").mean([1, 2, 3])
                ref_loss_l = F.mse_loss(ref_pred_l, target_l, reduction="none").mean([1, 2, 3])

                logits = (ref_loss_w - loss_w) - (ref_loss_l - loss_l)
                loss = -torch.nn.functional.logsigmoid(beta * logits).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                if global_step % 10 == 0:
                    print(f"Step {global_step} Loss: {loss.item():.4f}")
        
        self.model.save_pretrained(self.config['save_root'])

    def get_optimizer(self, params):
        lr = float(self.config['optimizer'].get('lr', 1e-6))
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr)
        except:
            return torch.optim.AdamW(params, lr=lr)