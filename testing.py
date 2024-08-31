import sys
import os

# Add YOLOv5 and DeepSORT to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'yolov5'))
sys.path.append(os.path.join(script_dir, 'deep_sort_files'))

import torch
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression
import nn_matching
from PIL import Image

print("PyTorch version:", torch.__version__)
print("OpenCV version:", cv2.__version__)
print("YOLOv5 and DeepSORT modules imported successfully")

# Try to load a YOLOv5 model
try:
    model = attempt_load('yolov5s.pt')
    print("YOLOv5 model loaded successfully")
except Exception as e:
    print("Error loading YOLOv5 model:", str(e))

# Try to create a NearestNeighborDistanceMetric
try:
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.3, None)
    print("NearestNeighborDistanceMetric created successfully")
except Exception as e:
    print("Error creating NearestNeighborDistanceMetric:", str(e))

# Test YOLOv5 inference
try:
    img = torch.zeros((1, 3, 640, 640))  # Example input
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred)
    print("YOLOv5 inference successful")
except Exception as e:
    print("Error during YOLOv5 inference:", str(e))