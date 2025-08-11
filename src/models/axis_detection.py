import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2

class AxisRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 4)

    def forward(self, x):
        return self.backbone(x)

class ShoeAxisDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = AxisRegressor().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def detect_axis(self, image):
        # Accepts either np.ndarray or PIL.Image.Image
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_pil = transforms.ToPILImage()(img_rgb)
        else:
            # Assume PIL.Image.Image
            img_pil = image
            w, h = img_pil.size
        img_tensor = self.transform(img_pil)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            axis = self.model(img_tensor).cpu().numpy().flatten()
        # axis is [x1_norm, y1_norm, x2_norm, y2_norm]
        x1, y1, x2, y2 = axis
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        return ((x1, y1), (x2, y2))
