import cv2
import os
from datetime import datetime
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "yolov8s.pt"
SAVE_DIR = r"C:\Users\vitta\OneDrive\Desktop\BagMatcherApp\bags\hand bag 300"
CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.4

# --- Setup ---
os.makedirs(SAVE_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)
model.verbose = False
cap = cv2.VideoCapture(CAMERA_ID)

print("📷 Bag detection started with YOLOv8s... Press 'q' to quit.")

# --- Real-time Detection Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, verbose=False)[0]
    class_names = model.names

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = class_names[cls_id]

        if label not in ['backpack', 'handbag', 'suitcase']:
            continue

        if conf >= CONFIDENCE_THRESHOLD:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"bag_{timestamp}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"👜 Bag detected at {timestamp} → Saved to {filepath}")
            cv2.waitKey(50)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup and Post-process ---
try:
    os.remove("yolov8s.pt")
    print("🗑️ yolov8s.pt deleted after execution.")
except Exception as e:
    print(f"⚠️ Could not delete yolov8s.pt: {e}")

cap.release()
cv2.destroyAllWindows()
