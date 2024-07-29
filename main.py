# Starter code

import torch
from ultralytics import YOLO
import cv2
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == "0":
    torch.cuda.set_device(0)

print(f'Using device: {device}')

url = 'http://129.161.161.235/stream'

model = YOLO("best.pt").to(device)

while (True):
    cap = cv2.VideoCapture(url)
    ret, im = cap.read()
    results = model.track(im, show=True)
