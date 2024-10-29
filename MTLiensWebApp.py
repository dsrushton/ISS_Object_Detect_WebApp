from flask import Flask, Response, render_template
import cv2
import threading
from flask_cors import CORS
import numpy as np
import torch
import subprocess
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import queue
import time
import os

app = Flask(__name__)
CORS(app)

# Constants
YOUTUBE_URL = 'https://www.youtube.com/watch?v=DfEr5XCFNWM'
MODEL_PATH = os.path.join('model', 'lowest_loss_faster_rcnn_model.pth')

# Global variables
frame_queue = queue.Queue(maxsize=10)
stop_event = threading.Event()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = T.Compose([T.ToTensor()])

class FastRCNNPredictor(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = torch.nn.Linear(in_channels, num_classes)
        self.bbox_pred = torch.nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

def get_best_stream_url():
    try:
        result = subprocess.run(
            ['yt-dlp', '-f', 'best', '-g', YOUTUBE_URL],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error fetching stream URL: {e.stderr}")
        return None

def get_stream_with_retry(max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        stream_url = get_best_stream_url()
        if stream_url:
            return stream_url
        print(f"Failed to get stream URL, attempt {attempt + 1}/{max_retries}")
        time.sleep(retry_delay)
    return None

def initialize_model():
    global model
    try:
        print(f"Initializing model from {MODEL_PATH}")
        model = fasterrcnn_resnet50_fpn(weights=None)
        num_classes = 11
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

class_colors = {
    'background': (255, 255, 255),
    'Space': (0, 0, 255),
    'Earth_broad': (0, 255, 0),
    'ISS': (255, 0, 0),
    'panels': (0, 255, 255),
    'e_t': (255, 0, 255),
    'no_feed': (255, 255, 0),
    't_d': (128, 0, 128),
    'lf': (255, 165, 0),
    'unknown': (128, 128, 128),
    'sun': (255, 255, 255)
}

class_thresholds = {
    'Space': 0.5,
    'Earth_broad': 0.5,
    'ISS': 0.5,
    'panels': 0.5,
    'e_t': 0.5,
    'no_feed': 0.5,
    't_d': 0.5,
    'lf': 0.5,
    'unknown': 0.1,
    'sun': 0.5,
}

def crop_frame(frame):
    left_crop = 165
    right_crop = 176
    cropped_frame = frame[:, left_crop:frame.shape[1] - right_crop]
    return cropped_frame

def process_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img)

    for element in predictions:
        boxes = element['boxes'].cpu().numpy().astype(np.int32)
        labels = element['labels'].cpu().numpy()
        scores = element['scores'].cpu().numpy()
        
        for i, box in enumerate(boxes):
            label = class_names[labels[i]]
            threshold = class_thresholds.get(label, 0.5)
            if scores[i] > threshold:
                color = class_colors.get(label, (0, 255, 0))
                label_position = (box[0], box[1] - 10) if box[1] > 20 else (box[0], box[1] + 20)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, f"{label} {scores[i]:.2f}", label_position, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

def stream_processor():
    while not stop_event.is_set():
        stream_url = get_stream_with_retry()
        if not stream_url:
            print("Failed to get stream URL after all retries")
            time.sleep(5)
            continue

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print("Failed to open video stream")
            time.sleep(5)
            continue

        try:
            consecutive_failures = 0
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    print(f"Failed to read frame. Failures: {consecutive_failures}")
                    if consecutive_failures > 5:
                        break
                    time.sleep(1)
                    continue

                frame = crop_frame(frame)
                processed_frame = process_frame(frame)

                if frame_queue.full():
                    frame_queue.get()
                frame_queue.put(processed_frame)

        finally:
            cap.release()

def generate_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    import os
    print(f"Current working directory: {os.getcwd()}")
    print(f"Templates directory exists: {os.path.exists('templates')}")
    print(f"Templates directory contents: {os.listdir('templates')}")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Initialize model and class names
    class_names = ['background', 'Space', 'Earth_broad', 'ISS', 'panels', 
                   'e_t', 'no_feed', 't_d', 'lf', 'unknown', 'sun']
    
    # Print GPU status
    if torch.cuda.is_available():
        print(f"GPU is available. Using {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available. Using CPU.")

    # Initialize model
    initialize_model()
    
    # Start stream processor in a separate thread
    processor_thread = threading.Thread(target=stream_processor)
    processor_thread.daemon = True
    processor_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)