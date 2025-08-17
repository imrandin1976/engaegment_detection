# utils/detect_engagement.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def predict_engagement(model, frame, device, class_names):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        label = class_names[pred.item()]
    return label
