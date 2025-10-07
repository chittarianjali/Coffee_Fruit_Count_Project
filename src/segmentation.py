import os
import csv
import cv2
from functools import lru_cache
from ultralytics import YOLO

# Path to your trained weights
MODEL_PATH = "runs/detect/train/weights/best.pt"  # update if needed

@lru_cache(maxsize=1)
def _get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at: {MODEL_PATH}")
    return YOLO(MODEL_PATH)

def segment_fruits(image_path: str, out_dir: str = "output", conf: float = 0.15, iou: float = 0.5):
    """
    Runs YOLOv8 on the given image, draws class-colored boxes,
    writes overlay + CSV, and returns (counts_dict, overlay_path).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    model = _get_model()
    results = model.predict(image_path, conf=conf, iou=iou, verbose=False)[0]
    names = results.names  # class id -> class name

    overlay = img.copy()
    counts = {"green": 0, "yellow": 0, "red": 0}

    for b in results.boxes:
        c = int(b.cls[0])
        label = names[c] if isinstance(names, dict) else names[c]
        confidence = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])

        color = (0, 255, 0) if label == "green" else (0, 255, 255) if label == "yellow" else (0, 0, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"{label} {confidence:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if label in counts:
            counts[label] += 1

    counts["total"] = sum(counts.values())

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    overlay_path = os.path.join(out_dir, f"{base}_overlay.jpg")
    cv2.imwrite(overlay_path, overlay)

    csv_path = os.path.join(out_dir, f"{base}_counts.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Green", "Yellow", "Red", "Total"])
        writer.writerow([counts["green"], counts["yellow"], counts["red"], counts["total"]])

    return counts, overlay_path
