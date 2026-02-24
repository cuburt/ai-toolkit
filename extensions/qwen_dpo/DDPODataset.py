import torch
from torch.utils.data import Dataset
import os

class DDPODataset(Dataset):
    def __init__(self, config):
        # This config['root_path'] should point to the OUTPUT_FOLDER from step 1
        self.root = config['root_path']
        self.files = [f for f in os.listdir(self.root) if f.endswith('.pt')]
        
        if len(self.files) == 0:
            print(f"WARNING: No .pt files found in {self.root}.")
            print("Did you run the preprocess.py script?")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the pre-computed tensors
        path = os.path.join(self.root, self.files[idx])
        data = torch.load(path, map_location='cpu', weights_only=True)
        
        # Remove batch dimensions usually added during inference [1, C, H, W] -> [C, H, W]
        return {
            "w_latents": data['w_latents'].squeeze(0),
            "l_latents": data['l_latents'].squeeze(0),
            "prompt_embeds": data['prompt_embeds'].squeeze(0),
            "pooled_embeds": data['pooled_embeds'].squeeze(0)
        }