import torch
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from UNetModel import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions,
)

# Hyperparameters
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  
IMAGE_WIDTH = 240  
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "dataset/train_images/"
TRAIN_MASK_DIR = "dataset/train_masks/"
VAL_IMG_DIR = "dataset/val_images/"
VAL_MASK_DIR = "dataset/val_masks/"
best_acc = -1

def train(loader, model, optim, loss_fn, scaler):
    loop = tqdm(loader)
    
    for batch_index, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
        predictions = model(data)
        if predictions.shape != targets.shape:
            targets = targets.view_as(predictions)
        loss = loss_fn(predictions, targets)
            
        optim.zero_grad()
        loss.backward()
        optim.step()
            
        loop.set_postfix(loss = loss.item())
        
train_transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
        )
        #transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
        )
        #transforms.ToTensor(),
    ]
)

model = UNet(in_channels=3, out_channels=1).to(DEVICE)

#cross entropy loss for multiple segmentation channels 
loss_fn = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loader, val_loader = get_loaders(
    TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, train_transforms, val_transforms,
    NUM_WORKERS, PIN_MEMORY
)

if LOAD_MODEL:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
check_accuracy(val_loader, model, device=DEVICE)


scaler = torch.GradScaler(DEVICE)

for epoch in range(NUM_EPOCHS):
    train(train_loader, model, optimizer, loss_fn, scaler)

    accuracy = check_accuracy(val_loader, model, device=DEVICE)
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)
    #check_accuracy(val_loader, model, device=DEVICE)
    save_predictions(val_loader, model, folder="saved_results", device=DEVICE)
    

def val_model(model, loader, device):
    model.eval()
    num_correct = 0
    num_samples = 0
    dice_score = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            if preds.shape != y.shape:
                y = y.view_as(preds)

            num_correct += (preds == y).sum().item()
            num_samples += preds.numel()
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
        
        accuracy = num_correct / num_samples
        avg_dice_score = dice_score / len(loader)

        print(f"Val Accuracy: {accuracy:.2f}")
        print(f"Val Dice Score: {avg_dice_score:.2f}")
    
    model.train()
    return accuracy, avg_dice_score
    

checkpoint_path = "my_checkpoint.pth.tar"

load_checkpoint(torch.load(checkpoint_path), model)
val_acc, val_dice = val_model(model, val_loader, device=DEVICE)
