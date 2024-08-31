import cv2
import torch
from yolov5 import YOLOv5
import numpy as np
from deep_sort.deep_sort import DeepSort

# Initialize YOLOv5 model for person detection
yolo_model = YOLOv5("yolov5s.pt", device="cuda" if torch.cuda.is_available() else "cpu")

# Initialize Deep SORT tracker
tracker = DeepSort(
    model_path="deep_sort/model_weights/ckpt.t7",  # Path to Deep SORT model weights
    max_age=30,
    min_hits=3,
    iou_threshold=0.3
)

def detect_and_classify_persons(frame, model):
    """
    Detect persons in the frame using YOLOv5 model and classify them as child or therapist.

    Args:
        frame (numpy.ndarray): The input video frame.
        model (YOLOv5): The YOLOv5 model for object detection.

    Returns:
        detections (list): List of detected bounding boxes and class labels.
    """
    # Perform detection
    results = model.predict(frame)

    # Filter detections for persons only
    detections = []
    for result in results:
        if result['label'] == 'person':  # Assuming YOLOv5 is trained to detect 'person' class
            bbox = result['box']  # [x1, y1, x2, y2]
            score = result['score']
            # TODO: Use additional classifier to distinguish between child and therapist
            label = 'child' if score > 0.5 else 'therapist'  # Placeholder logic for classification
            detections.append([*bbox, score, label])

    return detections

def update_tracker(frame, detections):
    """
    Update the Deep SORT tracker with new detections.

    Args:
        frame (numpy.ndarray): The input video frame.
        detections (list): List of detected bounding boxes and class labels.

    Returns:
        tracks (list): List of active tracks with IDs and associated detections.
    """
    # Convert detections to format compatible with Deep SORT
    bboxes = np.array([det[:4] for det in detections])
    scores = np.array([det[4] for det in detections])
    labels = np.array([det[5] for det in detections])

    # Update tracker
    tracker.predict()
    tracker.update(bboxes, scores, labels, frame)

    # Get active tracks
    tracks = tracker.get_tracked_objects()

    return tracks

def draw_annotations(frame, tracks):
    """
    Draw bounding boxes, labels, and IDs on the video frame.

    Args:
        frame (numpy.ndarray): The input video frame.
        tracks (list): List of active tracks with IDs.

    Returns:
        frame (numpy.ndarray): The annotated video frame.
    """
    for track in tracks:
        bbox = track['bbox']
        track_id = track['track_id']
        label = track['label']
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if label == 'child' else (255, 0, 0)

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            frame,
            f'{label} ID: {track_id}',
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    return frame

def process_video(video_path, output_path, model):
    """
    Process the input video, perform detection and tracking, and save the output video.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output video file.
        model (YOLOv5): The YOLOv5 model for object detection.
    """
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_and_classify_persons(frame, model)
        tracks = update_tracker(frame, detections)
        annotated_frame = draw_annotations(frame, tracks)
        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = 'input_video.mp4'  # Replace with the path to your input video
    output_video_path = 'output_video.mp4'  # Replace with the desired output video path

    # Run the pipeline
    process_video(input_video_path, output_video_path, yolo_model)
