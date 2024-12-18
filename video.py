# For use with HTTP stream from ESP32-WROVER camera module

import torch
from ultralytics import YOLO
import cv2
import sys

if __name__ == "__main__":
    url = sys.argv[1] if len (sys.argv) > 1 else 'http://129.161.161.235/stream'
    if not "http://" in url:
        url = "http://" + url

    if not "/stream" in url:
        url += "/stream"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "0":
        torch.cuda.set_device(0)

    print(f'Using device: \t{device}')
    print(f'At url: \t{url}')

    model = YOLO("best.pt").to(device)
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # TODO: Update object distance list
    objects = {
        'Plastic bottle cap':1,
        'Other plastic bottle': 8,
        'Clear plastic bottle': 8,
        'Glass bottle': 8,
        'Drink can': 3,
        'Paper straw': 8,
        'Battery': 2
    }

    while (True):
        ret, im = cap.read()
        results = model.track(im, show=True, conf=0.01)
        boxes = results[0].boxes
        for box in boxes.numpy():
            pos = box.xywh
            cls = box.cls
            x1 = pos[0][0]
            y1 = pos[0][1]
            w = pos[0][2]
            h = pos[0][3]
            name = model.names[int(cls)]
            D = -1
            actW = objects.get(name)
            if actW != None:
                D = (actW * 3650) / w   # TODO: update focal length

            print(f"Coordinates: {x1}, {y1}, {w}, {h}, Class: {name}, Distance away: {D}")
