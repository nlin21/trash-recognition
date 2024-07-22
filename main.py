# Starter code

import torch
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == "0":
    torch.cuda.set_device(0)

print(f'Using device: {device}')

model = YOLO("yolov8n.pt").to(device)
results = model.track(source=0, show=True, save=True)
