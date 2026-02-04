import torch
import torch.nn.functional as F
import copy
from toolkit.job import Process
from toolkit.train_tools import get_optimizer
from diffusers import FlowMatchEulerDiscreteScheduler, Transformer2DModel

# Import Qwen specific class if available, else generic DiT
try:
    from diffusers import QwenImageTransformer2DModel

    TransformerClass = QwenImageTransformer2DModel
except ImportError:
    TransformerClass = Transformer2DModel


class DDPOTrainer(Process):
    def __init__(self, process_id, job, config):
        super().__init__(process_id, job, config)
        self.device = torch.device(self.config.get('device', 'cuda'))
        self.dtype = torch.bfloat16 if self.config.get('dtype') == 'bf16' else torch.float16

    def load_model(self):
        # 1. Load Scheduler (Flow Matching for Qwen/SD3)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config['model']['name_or_path'],
            subfolder="scheduler"
        )

        # 2. Load Student Model
        print(f"Loading Student: {self.config['model']['name_or_path']}")
        self.model = TransformerClass.from_pretrained(
            self.config['model']['name_or_path'],
            subfolder="transformer",
            torch_dtype=self.dtype
        ).to(self.device)

        # Enable Gradient Checkpointing (Critical for DPO)
        self.model.enable_gradient_checkpointing()

        # 3. Load Reference Model (Frozen)
        print("Loading Reference Model...")
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()

    def get_dit_prediction(self, model, latents, t, prompt_embeds, pooled_embeds):
        """
        Qwen/MMDiT specific forward pass.
        Requires 'pooled_projections' (the pooled vector from Qwen2-VL).
        """
        return model(
            hidden_states=latents,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            return_dict=False
        )[0]

    def run(self):
        # Initialize
        self.load_model()

        # Dataset
        from .DDPODataset import DDPODataset
        dataset = DDPODataset(self.config['dataset'])
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['dataset'].get('batch_size', 1),
            shuffle=True,
            num_workers=2
        )

        # Optimizer
        optimizer = get_optimizer(self.model.parameters(), self.config['optimizer'])

        # Training Loop
        self.model.train()
        beta = self.config.get('dpo_beta', 2000)
        num_epochs = self.config.get('num_epochs', 1)

        print(f"Starting Training for {num_epochs} epochs...")

        global_step = 0
        for epoch in range(num_epochs):
            for step, batch in enumerate(dataloader):

                # --- 1. Prepare Inputs ---
                # Assume dataset yields latent tensors (pre-encoded) or we encode here.
                # For Qwen DPO, we strongly recommend pre-encoding to save VRAM.
                latents_w = batch['w_latents'].to(self.device, dtype=self.dtype)
                latents_l = batch['l_latents'].to(self.device, dtype=self.dtype)
                p_embeds = batch['prompt_embeds'].to(self.device, dtype=self.dtype)
                pool_embeds = batch['pooled_embeds'].to(self.device, dtype=self.dtype)

                bsz = latents_w.shape[0]

                # --- 2. Sample Noise & Time ---
                noise = torch.randn_like(latents_w)
                timesteps = torch.randint(0, 1000, (bsz,), device=self.device).long()

                # --- 3. Add Noise (Flow Matching) ---
                noisy_w = self.scheduler.add_noise(latents_w, noise, timesteps)
                noisy_l = self.scheduler.add_noise(latents_l, noise, timesteps)

                # --- 4. Student Forward ---
                pred_w = self.get_dit_prediction(self.model, noisy_w, timesteps, p_embeds, pool_embeds)
                pred_l = self.get_dit_prediction(self.model, noisy_l, timesteps, p_embeds, pool_embeds)

                # --- 5. Reference Forward (No Grad) ---
                with torch.no_grad():
                    ref_pred_w = self.get_dit_prediction(self.ref_model, noisy_w, timesteps, p_embeds, pool_embeds)
                    ref_pred_l = self.get_dit_prediction(self.ref_model, noisy_l, timesteps, p_embeds, pool_embeds)

                # --- 6. DPO Loss Calculation ---
                # Target for Flow Matching is usually (noise - latents) or velocity
                # We use the same target for both to compute reconstruction error
                target = noise - latents_w  # Simplified FM target

                # MSE "Energy"
                loss_w = F.mse_loss(pred_w, target, reduction="none").mean([1, 2, 3])
                loss_l = F.mse_loss(pred_l, target, reduction="none").mean([1, 2, 3])
                ref_loss_w = F.mse_loss(ref_pred_w, target, reduction="none").mean([1, 2, 3])
                ref_loss_l = F.mse_loss(ref_pred_l, target, reduction="none").mean([1, 2, 3])

                # Preference Logic: Maximize (reward_w - reward_l)
                # reward = ref_error - model_error
                logits = (ref_loss_w - loss_w) - (ref_loss_l - loss_l)

                loss = -torch.nn.functional.logsigmoid(beta * logits).mean()

                # --- 7. Backprop ---
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                # --- 8. Logging/Saving ---
                # Use toolkit's internal logging if available, or print
                if step % 10 == 0:
                    print(f"Step {global_step} [E{epoch}]: Loss {loss.item():.4f}")

                # Save checkpoint every X steps (Simple implementation)
                if global_step % self.config.get('save_steps', 500) == 0:
                    save_path = f"{self.config['save_root']}/step_{global_step}"
                    self.model.save_pretrained(save_path)
                    print(f"Saved checkpoint to {save_path}")