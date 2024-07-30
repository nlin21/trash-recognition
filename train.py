from ultralytics import YOLO
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "0":
        torch.cuda.set_device(0)

    print(f'Using device: {device}')

    model = YOLO("./models/yolov8s.pt").to(device)  
    model.train(data="data.yaml", epochs=100, patience=25, imgsz=640, pretrained=True, optimizer='SGD')