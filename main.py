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

# =====================
# FLASK INIT
# =====================
app = Flask(__name__)
sock = Sock(app)

# =====================
# YOLO INIT
# =====================
model = YOLO("yolov8n.pt")
model.fuse()

# =====================
# TTS SETUP
# =====================
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

# =====================
# DISTANCE SETUP
# =====================
FOCAL_LENGTH = 650
KNOWN_WIDTHS = {
    "person": 50,
    "chair": 45,
    "bottle": 7,
    "cup": 8,
    "cell phone": 7,
    "laptop": 30
}

def approx_distance(px, label):
    if px < 10:
        return -1
    w = KNOWN_WIDTHS.get(label.lower(), 20)
    return min(max((w * FOCAL_LENGTH) / px, 10), 400)

# =====================
# ROUTE
# =====================
@app.route("/")
def index():
    return send_file("index.html")

# =====================
# WEBSOCKET
# =====================
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

# =====================
# RUN
# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=False)
