#Modules and dependencies importing

import cv2
import numpy as np
import base64
import time
import queue
import threading

from flask import Flask, send_file
from flask_sock import Sock
from ultralytics import YOLO
import pyttsx3

# Initiation of Flask app and YOLO model
app = Flask(__name__)
sock = Sock(app)


model = YOLO("yolov8n.pt")
model.fuse()

tts_queue = queue.Queue(maxsize=3)
engine = pyttsx3.init()
engine.setProperty("rate", 145)

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass

threading.Thread(target=tts_worker, daemon=True).start()
#Known widths and focal length for distance approximation

FOCAL_LENGTH = 650
KNOWN_WIDTHS = {
    "person": 50,
    "bicycle": 60,
    "car": 180,
    "motorcycle": 80,
    "airplane": 3000,
    "bus": 250,
    "train": 300,
    "truck": 250,
    "boat": 200,
    "traffic light": 30,
    "fire hydrant": 30,
    "stop sign": 75,
    "parking meter": 30,
    "bench": 120,
    "bird": 25,
    "cat": 30,
    "dog": 50,
    "horse": 150,
    "sheep": 100,
    "cow": 150,
    "elephant": 350,
    "bear": 200,
    "zebra": 140,
    "giraffe": 180,
    "backpack": 35,
    "umbrella": 100,
    "handbag": 30,
    "tie": 10,
    "suitcase": 45,
    "frisbee": 25,
    "skis": 180,
    "snowboard": 160,
    "sports ball": 22,
    "kite": 60,
    "baseball bat": 7,
    "baseball glove": 25,
    "skateboard": 20,
    "surfboard": 50,
    "tennis racket": 27,
    "bottle": 7,
    "wine glass": 8,
    "cup": 8,
    "fork": 3,
    "knife": 3,
    "spoon": 3,
    "bowl": 20,
    "banana": 4,
    "apple": 8,
    "sandwich": 12,
    "orange": 8,
    "broccoli": 12,
    "carrot": 4,
    "hot dog": 5,
    "pizza": 30,
    "donut": 10,
    "cake": 30,
    "chair": 45,
    "couch": 180,
    "potted plant": 30,
    "bed": 160,
    "dining table": 150,
    "toilet": 50,
    "tv": 100,
    "laptop": 30,
    "mouse": 6,
    "remote": 5,
    "keyboard": 45,
    "cell phone": 7,
    "microwave": 50,
    "oven": 60,
    "toaster": 30,
    "sink": 60,
    "refrigerator": 70,
    "book": 15,
    "clock": 25,
    "vase": 20,
    "scissors": 8,
    "teddy bear": 30,
    "hair drier": 15,
    "toothbrush": 2
}


def approx_distance(px, label):
    if px < 10:
        return -1
    w = KNOWN_WIDTHS.get(label.lower(), 20)
    return min(max((w * FOCAL_LENGTH) / px, 10), 400)

# Flask route for index page
@app.route("/")
def index():
    return send_file("index.html")


last_dist = {}
last_time = {}
MIN_INTERVAL = 3.0

@sock.route("/ws")
def ws_handler(ws):
    while True:
        data = ws.receive()
        if data is None:
            break

        try:
            frame_bytes = base64.b64decode(data)
            frame_array = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        except:
            continue

        if frame is None:
            continue

        h, w, _ = frame.shape
        center = w // 2

        results = model(frame, conf=0.35, verbose=False)
# Processing detections and estimating distances
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]

                width_px = x2 - x1
                dist = approx_distance(width_px, label)
                if dist == -1 or dist > 250:
                    continue

                if x2 < center:
                    direction = "left"
                elif x1 > center:
                    direction = "right"
                else:
                    direction = "ahead"

                key = f"{label}_{x1//100}"
                now = time.time()

                if (key not in last_dist or abs(dist - last_dist[key]) > 15) and \
                   now - last_time.get(key, 0) > MIN_INTERVAL:

                    if not tts_queue.full():
                        tts_queue.put(f"{label} {direction}, {int(dist)} centimeters")

                    last_dist[key] = dist
                    last_time[key] = now

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f"{label} {direction} {int(dist)}cm",
                    (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,255),
                    2
                )

        _, buf = cv2.imencode(".jpg", frame)
        ws.send(base64.b64encode(buf).decode())
# Running the Flask app and Estalishing the ports
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=False)
