import base64
import time
import threading
import queue
import cv2
import numpy as np

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_sock import Sock

from ultralytics import YOLO
import pyttsx3

from simplify import simplify_text, highlight_difficult_words

# =====================
# FLASK
# =====================
app = Flask(__name__, static_folder="static", template_folder="templates")
sock = Sock(app)

# =====================
# YOLO
# =====================
MODEL_PATH = "yolov8n.pt"

model = YOLO(MODEL_PATH)
try:
    model.to("cuda")
except Exception:
    print("⚠️ CUDA not available, using CPU")

model.fuse()

# =====================
# TTS (pyttsx3 – OFFLINE)
# =====================
tts_q = queue.Queue(maxsize=10)

engine = pyttsx3.init()
engine.setProperty("rate", 145)
engine.setProperty("volume", 1.0)

voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)

def tts_worker():
    while True:
        text = tts_q.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("❌ TTS error:", e)

threading.Thread(target=tts_worker, daemon=True).start()

# =====================
# ROUTES
# =====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

# ---------------------
# Text Simplification
# ---------------------
@app.route("/simplify", methods=["POST"])
def simplify_api():
    data = request.get_json() or {}
    text = data.get("text", "")

    return jsonify({
        "simplified": simplify_text(text),
        "highlighted": highlight_difficult_words(text)
    })

# ---------------------
# Backend TTS endpoint
# ---------------------
@app.route("/tts", methods=["POST"])
def tts_api():
    data = request.get_json() or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"ok": False})

    try:
        tts_q.put_nowait(text)
        return jsonify({"ok": True})
    except queue.Full:
        return jsonify({"ok": False, "msg": "TTS queue full"}), 429

# =====================
# WEBSOCKET (Camera → YOLO → Browser)
# =====================
@sock.route("/ws")
def ws_handler(ws):
    print("✅ WebSocket connected")

    last_infer = 0
    INFER_INTERVAL = 0.2  # ~5 FPS inference

    while True:
        data = ws.receive()
        if data is None:
            print("❌ WebSocket closed")
            break

        try:
            frame_bytes = base64.b64decode(data)
            np_img = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        except Exception:
            continue

        if frame is None:
            continue

        now = time.time()
        if now - last_infer < INFER_INTERVAL:
            continue
        last_infer = now

        try:
            results = model(frame, conf=0.35, verbose=False)
        except Exception as e:
            print("❌ YOLO error:", e)
            continue

        h, w = frame.shape[:2]
        center = w // 2

        for r in results:
            for box in r.boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    direction = (
                        "left" if x2 < center else
                        "right" if x1 > center else
                        "ahead"
                    )

                    text = f"{label} {direction}"

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(
                        frame, text,
                        (x1, max(y1-10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,0), 2
                    )
                except Exception:
                    continue

        try:
            _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            ws.send(base64.b64encode(jpg).decode("utf-8"))
        except Exception:
            pass

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
