# main.py

import cv2
import torch
import numpy as np
from deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import os
import argparse

class YOLOModel:
    def __init__(self, weights_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        self.model.eval()

    def detect(self, frame):
        results = self.model(frame)
        return results.xyxy[0].cpu().numpy()

class ClassifierModel:
    def __init__(self, weights_path):
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)  # 2 classes: adult and child
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classify(self, crop):
        crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop = self.transform(crop).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(crop)
        
        class_id = torch.argmax(output).item()
        return "Child" if class_id == 1 else "Adult"

class PersonDetectionTrackingPipeline:
    def __init__(self, yolo_weights, classifier_weights, deepsort_weights, max_cosine_distance=0.3, nn_budget=None, max_iou_distance=0.7, max_age=30):
        self.yolo_model = YOLOModel(yolo_weights)
        self.classifier_model = ClassifierModel(classifier_weights)
        
        max_cosine_distance = max_cosine_distance
        nn_budget = nn_budget
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age)

        self.encoder = gdet.create_box_encoder(deepsort_weights, batch_size=1)
        self.track_history = {}

    def detect_and_track(self, frame):
        # YOLO detection
        boxes = self.yolo_model.detect(frame)
        
        # Filter for person class (assuming class 0 is person in YOLO)
        person_boxes = boxes[boxes[:, -1] == 0]

        # Classification
        classifications = []
        for box in person_boxes:
            crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            classification = self.classifier_model.classify(crop)
            classifications.append(classification)

        # Prepare detections for DeepSORT
        features = self.encoder(frame, person_boxes)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(person_boxes, features)]

        # Update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # Prepare results
        results = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            
            # Assign classification based on track history or current detection
            if track.track_id in self.track_history:
                class_name = self.track_history[track.track_id]
            else:
                class_name = classifications[len(classifications) - 1] if track.track_id >= len(classifications) else classifications[track.track_id - 1]
                self.track_history[track.track_id] = class_name
            
            results.append((bbox, track.track_id, class_name))

        return results

    def draw_boxes(self, frame, results):
        for bbox, track_id, class_name in results:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0) if class_name == "Child" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name}-{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.detect_and_track(frame)
            frame_with_boxes = self.draw_boxes(frame, results)

            out.write(frame_with_boxes)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")

        cap.release()
        out.release()
        print(f"Video processing complete. Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Person Detection and Tracking Pipeline")
    parser.add_argument("--input_video", required=True, help="Path to input video")
    parser.add_argument("--output_video", required=True, help="Path to output video")

    args = parser.parse_args()

    # Specify the paths to your model weights here
    yolo_weights = r"C:\Users\BQ Team 4\Documents\child\yolov5s.pt"
    classifier_weights = r"C:\Users\BQ Team 4\Documents\child\classifier_weights.pt"
    deepsort_weights = r"C:\Users\BQ Team 4\Documents\child\mars-small128.pb"

    pipeline = PersonDetectionTrackingPipeline(
        yolo_weights,
        classifier_weights,
        deepsort_weights
    )
    pipeline.process_video(args.input_video, args.output_video)

if __name__ == "__main__":
    main()