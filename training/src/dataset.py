import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class KonIQDataset(Dataset):
    def __init__(self, csv_path, img_dir, split="training"):
        df = pd.read_csv(csv_path)
        self.df = df[df["set"] == split].reset_index(drop=True)
        self.img_dir = img_dir

        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"])

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        mos = row["MOS"] / 100.0  # normalize to [0,1]

        return image, torch.tensor([mos], dtype=torch.float32)
