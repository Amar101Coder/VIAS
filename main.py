import cv2
import numpy as np
from flask import Flask, send_file
from flask_sock import Sock
from ultralytics import YOLO
import pyttsx3
import base64
import threading
import queue
import time

# =======================
# APP INIT
# =======================
app = Flask(__name__)
sock = Sock(app)

# =======================
# YOLO INIT
# =======================
model = YOLO("yolov8n.pt")
model.fuse()

# =======================
# TTS SETUP
# =======================
tts_queue = queue.Queue(maxsize=3)
engine = pyttsx3.init()
engine.setProperty("rate", 150)

def tts_player():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass

threading.Thread(target=tts_player, daemon=True).start()

# =======================
# DISTANCE ESTIMATION
# =======================
FOCAL_LENGTH = 650  # calibrate later

# =======================
# COCO CLASSES WIDTHS (cm)
# =======================
KNOWN_WIDTHS = {
    "person": 50,
    "bicycle": 170,
    "car": 180,
    "motorcycle": 80,
    "airplane": 3500,
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
    "dog": 40,
    "horse": 80,
    "sheep": 60,
    "cow": 90,
    "elephant": 250,
    "bear": 120,
    "zebra": 80,
    "giraffe": 120,
    "backpack": 35,
    "umbrella": 100,
    "handbag": 30,
    "tie": 10,
    "suitcase": 45,
    "frisbee": 25,
    "skis": 10,
    "snowboard": 25,
    "sports ball": 22,
    "kite": 100,
    "baseball bat": 7,
    "baseball glove": 30,
    "skateboard": 20,
    "surfboard": 60,
    "tennis racket": 30,
    "bottle": 7,
    "wine glass": 8,
    "cup": 8,
    "fork": 3,
    "knife": 3,
    "spoon": 3,
    "bowl": 15,
    "banana": 5,
    "apple": 8,
    "sandwich": 10,
    "orange": 8,
    "broccoli": 15,
    "carrot": 3,
    "hot dog": 5,
    "pizza": 30,
    "donut": 10,
    "cake": 30,
    "chair": 45,
    "couch": 200,
    "potted plant": 30,
    "bed": 160,
    "dining table": 150,
    "toilet": 40,
    "tv": 90,
    "laptop": 30,
    "mouse": 6,
    "remote": 5,
    "keyboard": 45,
    "cell phone": 7,
    "microwave": 50,
    "oven": 60,
    "toaster": 30,
    "sink": 50,
    "refrigerator": 70,
    "book": 15,
    "clock": 30,
    "vase": 20,
    "scissors": 8,
    "teddy bear": 30,
    "hair drier": 15,
    "toothbrush": 2
}

def approx_distance(pixel_width, label):
    if pixel_width < 10:
        return -1
    real_width = KNOWN_WIDTHS.get(label.lower(), 20)
    dist = (real_width * FOCAL_LENGTH) / pixel_width
    return min(max(dist, 10), 500)

# =======================
# ROUTES
# =======================
@app.route("/")
def index():
    return send_file("index.html")

# =======================
# WEBSOCKET
# =======================
last_distances = {}
last_tts_time = {}
MIN_TTS_INTERVAL = 1.5

@sock.route("/ws")
def yolo_ws(ws):
    while True:
        try:
            frame_data = ws.receive()
            if frame_data is None:
                break
            if len(frame_data) < 1000:
                continue

            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w, _ = frame.shape
            frame_center = w // 2

            results = model(frame, verbose=False, conf=0.4)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                    width_px = x2 - x1
                    distance = approx_distance(width_px, label)
                    if distance == -1:
                        continue

                    if x2 < frame_center:
                        direction = "left"
                    elif x1 > frame_center:
                        direction = "right"
                    else:
                        direction = "ahead"

                    key = f"{label}_{x1//100}"
                    now = time.time()

                    if (
                        key not in last_distances
                        or abs(distance - last_distances[key]) > 7
                    ) and now - last_tts_time.get(key, 0) > MIN_TTS_INTERVAL:

                        if not tts_queue.full():
                            tts_queue.put(
                                f"{label} {direction}, approximately {distance:.0f} centimeters away"
                            )
                        last_distances[key] = distance
                        last_tts_time[key] = now

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} | {direction} | {distance:.0f}cm",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2
                    )

            _, buffer = cv2.imencode(".jpg", frame)
            ws.send(base64.b64encode(buffer).decode("utf-8"))

        except Exception as e:
            print("WebSocket error:", e)
            break

# =======================
# RUN
# =======================
if __name__ == "__main__":
    app.run("0.0.0.0", 5000, debug=False)
