import os
import json
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim

class ShoeAxisDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        self.samples = []
        self.img_size = img_size
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
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, axis = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        axis = np.array(axis).reshape(-1, 2)  # [[x1, y1], [x2, y2]]
        axis = axis / np.array([[w, h]])  # normalize to [0,1]
        axis = axis.flatten()
        img = self.transform(img)
        return img, torch.tensor(axis, dtype=torch.float32)

class AxisRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 4)

    def forward(self, x):
        return self.backbone(x)

def train_model(data_dir, epochs=20, batch_size=16, lr=1e-4, img_size=224, save_path='trained_models/axis_detection/resnet50_axis.pth'):
    dataset = ShoeAxisDataset(data_dir, img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = AxisRegressor().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, targets in tqdm(loader):
            imgs, targets = imgs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataset):.4f}")
        torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='marked_data')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--save_path', type=str, default='trained_models/axis_detection/resnet50_axis.pth')
    args = parser.parse_args()
    train_model(args.data_dir, args.epochs, args.batch_size, args.lr, args.img_size, args.save_path)
