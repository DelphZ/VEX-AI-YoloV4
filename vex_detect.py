import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *
import traceback2 as traceback

class YOLODetector:
    def __init__(self, weights, cfg, names, img_size=(480, 640), conf_thres=0.5):  # Changed to tuple (height, width)
        self.device = select_device('')
        self.img_h, self.img_w = img_size  # Height first for OpenCV compatibility
        self.model = Darknet(cfg, (self.img_h, self.img_w)).to(self.device)  # Pass tuple
        self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
        self.model.eval()
        self.names = self.load_classes(names)
        self.conf_thres = conf_thres
        print("yolo initialized finished")
    
    def load_classes(self, path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        print("load class finished")
        return list(filter(None, names)) 

    def detect(self, image):
        #print("start detecting")
        img = self.preprocess(image)
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, self.conf_thres, 0.45)
        return self.postprocess(pred, image.shape)

    def preprocess(self, image):
        # Maintain aspect ratio with letterbox
        img = letterbox(image, new_shape=(self.img_h, self.img_w))[0]  # Use tuple
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img).to(self.device).unsqueeze(0)

    def postprocess(self, pred, orig_shape):
        detections = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                try:
                    # Get proper shapes
                    model_input_shape = (self.img_h, self.img_w)
                    orig_hw = orig_shape[:2]  # (height, width)

                    # Scale coordinates
                    scaled_coords = scale_coords(
                        model_input_shape,  # (height, width)
                        det[:, :4],         # Raw coordinates
                        orig_hw             # Original (height, width)
                    ).round()

                    # Combine coordinates with conf/cls
                    processed_det = torch.cat((
                        scaled_coords,
                        det[:, 4:]
                    ), dim=1)

                    # Convert to numpy
                    det_np = processed_det.cpu().numpy()

                    for row in det_np:
                        x1, y1, x2, y2, conf, cls = row
                        detections.append({
                            'class': int(cls),
                            'confidence': float(conf),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })
                        #print(f"class obj {int(cls)} appended")
                    
                except Exception as e:
                    print(f"Error during processing: {str(e)}")
                    traceback.print_exc()

        return detections
