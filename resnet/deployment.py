import os
import sys
import torch
import cv2
import numpy as np
import torchvision

sys.path.append('../third_party/torch2trt')
from torch2trt import TRTModule

device = torch.device('cuda')
mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)


def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x

if __name__=='__main__':

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load('resnet18_trt.pth'))

    video_path = '../../CCTV/videos/WildTrack/cam3_short.mp4'
    assert os.path.isfile(video_path)
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.resize(frame, (224, 224))
        output = model_trt(preprocess(image)).detach().cpu().numpy().flatten()
        idx = output.argmax()
        print(idx)