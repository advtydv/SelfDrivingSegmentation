import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_dir, train_mask_dir, val_dir, val_mask_dir, batch, train_transform, val_transform, num_workers=4, pin_memory=True):

    train_dataset = CarvanaDataset(train_dir, train_mask_dir, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    val_dataset = CarvanaDataset(val_dir, val_mask_dir, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader

def check_accuracy(loader, model, device="mps"):
    correct = 0
    pixels = 0
    dice_score = 0
    
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            predictions = torch.sigmoid(model(x))
            predictions = (predictions > 0.5).float()

            if predictions.shape != y.shape:
                y = y.view_as(predictions)

            correct += (predictions == y).sum().item()
            pixels += y.numel()
            dice_score += (2 * (predictions*y).sum()) / ((predictions + y).sum() + 1e-8)

        accuracy = correct/pixels
        avg_dice_score = dice_score/len(loader)

        print(f"got {correct}/{pixels} with accuracy {accuracy*100:.3f}")
        print(f"Dice score: {avg_dice_score}")

        model.train()
        return accuracy

def save_predictions(loader, model, folder="saved_results", device="mps"):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            predictions = torch.sigmoid(model(x))
            predictions = (predictions > 0.5).float()

        for i in range(predictions.shape[0]):
            torchvision.utils.save_image(predictions[i], f"{folder}/pred_{idx}_{i}.png")
            torchvision.utils.save_image(y[i], f"{folder}/gt_{idx}_{i}.png")

    model.train()

