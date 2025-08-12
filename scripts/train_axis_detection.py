import os
import json
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

class ShoeAxisDataset(Dataset):
    def __init__(self, root_dir, img_size=224, train=True):
        self.samples = []
        self.img_size = img_size
        self.train = train
        for model_dir in os.listdir(root_dir):
            model_path = os.path.join(root_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
            for annotator in os.listdir(model_path):
                annotator_path = os.path.join(model_path, annotator)
                if not os.path.isdir(annotator_path):
                    continue
                jpg_files = [f for f in os.listdir(annotator_path) if f.endswith('.JPG')]
                for fname in jpg_files:
                    img_path = os.path.join(annotator_path, fname)
                    json_path = img_path + '.json'
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            json_data = json.load(f)
                        # Find typeId for LongitudinalAxis
                        axis_type_id = None
                        for t in json_data.get('metadata', {}).get('types', []):
                            if t.get('name') == 'LongitudinalAxis':
                                axis_type_id = t.get('id')
                                break
                        axis = None
                        if axis_type_id:
                            for m in json_data.get('data', {}).get('markings', []):
                                if m.get('typeId') == axis_type_id and m.get('markingClass') == 'line_segment':
                                    origin = m.get('origin')
                                    endpoint = m.get('endpoint')
                                    if origin and endpoint:
                                        axis = [ [origin['x'], origin['y']], [endpoint['x'], endpoint['y']] ]
                                        break
                        if axis and len(axis) == 2:
                            self.samples.append((img_path, axis))
                            print(f"Found: {img_path} with axis annotation.")
                        else:
                            print(f"Skipped (no axis or wrong format): {json_path}")
                    else:
                        print(f"Skipped (no json): {img_path}")
        if self.train:
            self.transform = A.Compose([
                A.Rotate(limit=10, p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, axis = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or cannot be loaded: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        # axis: [[x1, y1], [x2, y2]]
        axis = np.array(axis).reshape(-1, 2)
        # Apply albumentations transform to both image and axis
        transformed = self.transform(image=img, keypoints=axis)
        img = transformed['image']
        axis_trans = np.array(transformed['keypoints'])
        # Normalize axis coordinates to [0, 1] after resizing
        axis_trans = axis_trans / np.array([[self.img_size, self.img_size]])
        axis_trans = axis_trans.flatten()
        return img, torch.tensor(axis_trans, dtype=torch.float32)

class AxisRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 4)

    def forward(self, x):
        return self.backbone(x)

def train_model(data_dir, epochs=20, batch_size=16, lr=1e-4, img_size=224, save_path='trained_models/axis_detection/resnet50_axis.pth'):
    from torch.utils.data import random_split
    import matplotlib.pyplot as plt

    dataset = ShoeAxisDataset(data_dir, img_size)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    model = AxisRegressor().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = -1
    patience = 10  # Early stopping patience
    epochs_since_improvement = 0
    best_model_wts = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            imgs, targets = imgs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_set)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                imgs, targets = imgs.cuda(), targets.cuda()
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * imgs.size(0)
        val_loss = val_running_loss / len(val_set)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_wts = model.state_dict()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
                break

    # Plot losses
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('axis_training_loss.png')
    plt.show()
    # Save best model
    if best_model_wts is not None:
        torch.save(best_model_wts, save_path)
        print(f"Best model saved to {save_path} (val loss: {best_val_loss:.4f})")
    else:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    # Load config.yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    axis_cfg = config['models']['axis_detection']
    save_path = config.get('axis_model', 'trained_models/axis_detection/resnet50_axis.pth')
    data_dir = args.data_dir if args.data_dir else config['paths']['raw_data']
    epochs = args.epochs if args.epochs else axis_cfg['epochs']
    batch_size = args.batch_size if args.batch_size else axis_cfg['batch_size']
    lr = args.lr if args.lr else axis_cfg['lr']
    img_size = args.img_size if args.img_size else axis_cfg['img_size']
    save_path = args.save_path if args.save_path else save_path

    train_model(data_dir, epochs, batch_size, lr, img_size, save_path)
