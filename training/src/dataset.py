import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class KonIQDataset(Dataset):
    def __init__(self, csv_path, img_dir, split="training"):
        assert split in ["training", "validation", "test"], f"Invalid split: {split}"
        df = pd.read_csv(csv_path)
        self.df = df[df["set"] == split].reset_index(drop=True)
        self.img_dir = img_dir

        # ImageNet normalization for pretrained backbones
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # KonIQ images are natively 512x384 (H,W)
        # Keep aspect ratio to avoid distortion
        if split == "training":
            self.transform = transforms.Compose([
                transforms.Resize((512, 384)),
                transforms.RandomCrop((384, 384)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 384)),
                transforms.CenterCrop((384, 384)),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # MOS is 0–100 in KonIQ → normalize to [0,1]
        mos_value = row["MOS"] / 100.0

        mos = torch.tensor(mos_value, dtype=torch.float32)  # Scalar

        return image, mos
