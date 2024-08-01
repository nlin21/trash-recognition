# For use with HTTP stream from ESP32-WROVER camera module
# 
# TODO: Look into converting to RTSP stream for direct integration with ultralyitcs.
# Right now, cv2 is used to capture and infer on each frame

import torch
from ultralytics import YOLO
import cv2

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

    while (True):
        ret, im = cap.read()
        results = model.track(im, show=True)