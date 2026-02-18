import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split



trainTransform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) 
])
testTransform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


class BelkiDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = row["imagePath"]
        label = row["label"]

        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
