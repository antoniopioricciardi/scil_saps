# dataset.py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import pickle
import numpy as np
from PIL import Image
import glob

class MarioSCILDataset(Dataset):
    def __init__(self, pkl_files, img_size=84, use_imagenet_norm=False):
        """
        Args:
            pkl_files: Can be:
                - Single string path: "mario_1_1.pkl"
                - List of paths: ["mario_1_1.pkl", "mario_1_2.pkl"]
                - Glob pattern: "mario_*.pkl" (loads all matching files)
            img_size: Image size to resize to (84 for Nature CNN, 224 for ResNet)
            use_imagenet_norm: Use ImageNet normalization (for pretrained ResNet) or [-1,1] (for from-scratch)
        """
        # Handle different input types
        if isinstance(pkl_files, str):
            if '*' in pkl_files or '?' in pkl_files:
                # Glob pattern
                pkl_files = sorted(glob.glob(pkl_files))
            else:
                # Single file
                pkl_files = [pkl_files]

        if not pkl_files:
            raise ValueError("No pickle files found!")

        print(f"Loading {len(pkl_files)} file(s)...")
        self.data = []

        for pkl_file in pkl_files:
            print(f"  Loading {pkl_file}...")
            with open(pkl_file, "rb") as f:
                file_data = pickle.load(f)
                self.data.extend(file_data)
                print(f"    Added {len(file_data)} frames")

        print(f"Total frames loaded: {len(self.data)}")

        # Preprocessing - Keep RGB for Mario (color is important!)
        transforms = [
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),  # Scales [0, 255] -> [0.0, 1.0]
        ]

        if use_imagenet_norm:
            # ImageNet normalization for pretrained ResNet
            transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        else:
            # Standard [-1, 1] normalization for training from scratch
            transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        self.transform = T.Compose(transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert raw numpy obs to processed tensor
        obs = self.transform(item['obs'])
        action = torch.tensor(item['action'], dtype=torch.long)
        
        return obs, action