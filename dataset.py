import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        #self.images = os.listdir(image_dir)
        self.images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        #print(f"Image shape: {image.shape}")
        #print(f"Mask shape: {mask.shape}")

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float()
        mask[mask == 255.0] = 1.0

        return image, mask
    

