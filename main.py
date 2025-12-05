import cv2
import numpy as np
from flask import Flask, send_file
from flask_sock import Sock
from ultralytics import YOLO
import pyttsx3
import base64
import threading
import queue

# -----------------------
# INIT
# -----------------------
app = Flask(__name__)
sock = Sock(app)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

model = YOLO("yolov8n.pt")  # YOLOv8 tiny model

# TTS
tts_queue = queue.Queue()
engine = pyttsx3.init()
engine.setProperty("rate", 150)

def tts_player():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

threading.Thread(target=tts_player, daemon=True).start()

# -----------------------
# DISTANCE SETUP
# -----------------------
FOCAL_LENGTH = 650  # adjust after calibration

KNOWN_WIDTHS = {
    "person":50.0, "bottle":7.0, "chair":40.0, "laptop":30.0, "cup":6.0
    # Add more classes as needed
}

def approx_distance(pixel_width, obj_name):
    if pixel_width <= 0:
        return -1
    width_cm = KNOWN_WIDTHS.get(obj_name.lower(), 10)  # default width
    return (width_cm * FOCAL_LENGTH) / pixel_width

# -----------------------
# ROUTES
# -----------------------
@app.route("/")
def index():
    return send_file("index.html")

# -----------------------
# WEBSOCKET
# -----------------------
last_distances = {}

@sock.route("/ws")
def yolo_ws(ws):
    global last_distances
    while True:
        try:
            frame_data = ws.receive()
            if frame_data is None:
                break

            # Convert bytes to image
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # YOLO detection
            results = model(frame, stream=True)
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                for box, cls_id, score in zip(boxes, classes, scores):
                    x1, y1, x2, y2 = map(int, box)
                    label = model.names[int(cls_id)]
                    w = x2 - x1

                    # Draw box & label
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,f"{label} {score:.2f}",(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

                    # Distance
                    distance = approx_distance(w, label)
                    cv2.putText(frame,f"{distance:.1f} cm",(x1,y2+20),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,200,0),2)

                    # TTS if distance changed
                    if abs(distance - last_distances.get(label,0)) > 5:
                        tts_queue.put(f"{label} is approximately {distance:.1f} centimeters away")
                        last_distances[label] = distance

            # Encode frame to JPEG
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = base64.b64encode(buffer).decode("utf-8")
            ws.send(frame_bytes)

        except Exception as e:
            print("WebSocket error:", e)
            break

# -----------------------
# RUN APP
# -----------------------
if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
