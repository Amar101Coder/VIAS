import cv2
from ultralytics import YOLO

# -----------------------
# CAMERA
# -----------------------
cap = cv2.VideoCapture(0)

# -----------------------
# LOAD YOLOv8 MODEL
# -----------------------
model = YOLO("yolov8n.pt")  # tiny model for speed

# -----------------------
# MAIN LOOP
# -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # YOLOv8 expects RGB
    results = model(frame, stream=True)  

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # bounding boxes
        scores = r.boxes.conf.cpu().numpy() # confidence
        classes = r.boxes.cls.cpu().numpy() # class ids

        for box, score, cls_id in zip(boxes, scores, classes):
            x1,y1,x2,y2 = map(int, box)
            label = model.names[int(cls_id)]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)

    cv2.imshow("YOLOv8 Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
