# For use with sample images - demo

import torch
from ultralytics import YOLO
import cv2

if __name__ == "__main__":
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if device == "0":
		torch.cuda.set_device(0)
          
	print(f'Using device: \t{device}')
	model = YOLO("best.pt").to(device)
	results = model.track("samples/sample3.png", show=True, save=True, conf=0.03)
	boxes = results[0].boxes
    
	for box in boxes.numpy():
		pos = box.xywh
		cls = box.cls
		x1 = pos[0][0]
		y1 = pos[0][1]
		w = pos[0][2]
		h = pos[0][3]
		name = model.names[int(cls)]
		print(f"Coordinates: {x1}, {y1}, {w}, {h}, Class: {name}")

