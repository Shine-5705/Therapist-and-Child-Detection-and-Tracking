import cv2
import torch
import numpy as np
from yolov5 import YOLOv5
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from torchvision import models, transforms
from PIL import Image

# Load YOLOv5 model
yolo = YOLOv5("yolov5s.pt")
yolo.conf = 0.25  # NMS confidence threshold
yolo.iou = 0.45  # NMS IoU threshold
yolo.classes = [0]  # Only detect persons (class 0 in COCO dataset)

# Load and fine-tune ResNet model for child/adult classification
resnet = models.resnet18(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes: child and adult
resnet.load_state_dict(torch.load('child_adult_classifier.pth'))
resnet.eval()

# Initialize DeepSORT
max_cosine_distance = 0.3
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Image transform for ResNet
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_frame(frame):
    # Detect persons using YOLOv5
    results = yolo(frame)
    boxes = results.pred[0][:, :4].cpu().numpy()
    confidences = results.pred[0][:, 4].cpu().numpy()
    
    # Classify each detection as child or adult
    classifications = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        out = resnet(batch_t)
        _, index = torch.max(out, 1)
        classifications.append("Child" if index == 0 else "Adult")
    
    # Prepare detections for DeepSORT
    detections = [Detection(box, confidence) for box, confidence in zip(boxes, confidences)]
    
    # Update tracker
    tracker.predict()
    tracker.update(detections)
    
    # Draw results
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = classifications[track.track_id % len(classifications)]
        color = (0, 255, 0) if class_name == "Child" else (0, 0, 255)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.putText(frame, f"{class_name}-{track.track_id}", (int(bbox[0]), int(bbox[1])-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame)
        out.write(processed_frame)
    
    cap.release()
    out.release()

# Example usage
process_video('input_video.mp4', 'output_video.mp4')