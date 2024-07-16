import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms.v2 as transforms
import re
import glob
from sklearn.model_selection import train_test_split
import scipy
from torch.utils.data import DataLoader

class KITTIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_files, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = image_files
        self.background_color = np.array([255, 0, 0])
        self.road_color = np.array([255, 0, 255]) 
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_file = self.images[index]
        image_path = os.path.join(self.image_dir, image_file)
        
        # Construct mask filename
        components = image_file.split('_')
        mask_file = f"{components[0]}_road_{components[1]}"  
        mask_path = os.path.join(self.mask_dir, mask_file)
        
        # Check if files exist
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        mask_np = np.array(mask)
        unique_colors = np.unique(mask_np.reshape(-1, 3), axis=0)
        print(f"Unique colors before in mask: {unique_colors}")

        tensor_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.3509, 0.3773, 0.3662],
                std=[0.2796, 0.2989, 0.3114],
            )
        ])

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = tensor_transform(image)

        mask_np = np.array(mask)
        unique_colors = np.unique(mask_np.reshape(-1, 3), axis=0)
        print(f"Unique colors after in mask: {unique_colors}")

        mask = np.all(mask_np == self.road_color[:,None,None], axis=0)
        mask = mask.astype(np.float32)  # Convert boolean to float32
        mask = torch.from_numpy(mask)

        #print(f"Mask shape: {mask.shape}")
        #print(f"Unique values in mask: {torch.unique(mask)}")
        #print(f"Percentage of road pixels: {mask.mean().item() * 100:.2f}%")

        return image, mask.unsqueeze(0)
    

    
def get_loaders(image_dir, mask_dir, batch, train_transform, val_transform, val_split=0.2, num_workers=0, pin_memory=True):
        all_images = [img for img in os.listdir(image_dir) if img.endswith(".png")]
        
        train_size = int(0.8 * len(all_images))
        val_size = len(all_images) - train_size
        train_images, val_images = torch.utils.data.random_split(all_images, [train_size, val_size])

        train_dataset = KITTIDataset(image_dir=image_dir, mask_dir=mask_dir, image_files=train_images, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

        val_dataset = KITTIDataset(image_dir=image_dir, mask_dir=mask_dir, image_files=val_images, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

        return train_loader, val_loader