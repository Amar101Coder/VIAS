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
# FLASK
# =====================
app = Flask(__name__)
sock = Sock(app)

# =====================
# YOLO
# =====================
model = YOLO("yolov8n.pt")
model.to("cuda")
model.fuse()

# =====================
# TTS
# =====================
tts_q = queue.Queue(maxsize=3)
engine = pyttsx3.init()
engine.setProperty("rate", 145)

def tts_worker():
    while True:
        text = tts_q.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

threading.Thread(target=tts_worker, daemon=True).start()

# =====================
# DISTANCE
# =====================
FOCAL_LENGTH = 650
KNOWN_WIDTHS = {
    "person": 50,
    "chair": 45,
    "bottle": 7,
    "cup": 8,
    "cell phone": 7,
    "laptop": 30,
}

def approx_distance(px, label):
    if px < 10:
        return -1
    w = KNOWN_WIDTHS.get(label.lower(), 20)
    return min(max((w * FOCAL_LENGTH) / px, 10), 400)

# =====================
# ROUTES
# =====================
@app.route("/")
def index():
    return send_file("index.html")

# =====================
# WEBSOCKET
# =====================
@sock.route("/ws")
def ws_handler(ws):
    print("✅ WebSocket connected")

    last_infer = 0
    INFER_INTERVAL = 0.20  # 5 FPS (safe)

    while True:
        data = ws.receive()
        if data is None:
            print("❌ WebSocket closed")
            break

        # Decode
        try:
            frame_bytes = base64.b64decode(data)
            np_img = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        except Exception as e:
            print("❌ Decode error", e)
            continue

        if frame is None:
            continue

        now = time.time()
        if now - last_infer < INFER_INTERVAL:
            continue
        last_infer = now

        print("🧠 Running YOLO")

        results = model(frame, conf=0.4, verbose=False)

        h, w, _ = frame.shape
        center = w // 2

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]

                width_px = x2 - x1
                dist = approx_distance(width_px, label)
                if dist == -1:
                    continue

                direction = (
                    "left" if x2 < center else
                    "right" if x1 > center else
                    "ahead"
                )

                txt = f"{label} {direction} {int(dist)}cm"

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, txt, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,255), 2)

        _, buf = cv2.imencode(".jpg", frame)
        ws.send(base64.b64encode(buf).decode("utf-8"))


# =====================
# RUN
# =====================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8001,
        debug=False,
        threaded=True,
        use_reloader=False
)

