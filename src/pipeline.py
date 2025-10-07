import os
import csv
import cv2
from ultralytics import YOLO

# -----------------------------
# Load YOLO Model (global so it loads only once)
# -----------------------------
MODEL_PATH = "runs/detect/train/weights/best.pt"  # update if needed
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO weights not found at {MODEL_PATH}")
model = YOLO(MODEL_PATH)


# -----------------------------
# Save Results (Overlay + CSV)
# -----------------------------
def _save_results(image_path, img, detections, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    overlay = img.copy()
    counts = {"green": 0, "yellow": 0, "red": 0}

    for box in detections.boxes:
        cls = int(box.cls[0])         # class index
        label = detections.names[cls] # 'green', 'yellow', 'red'
        conf = float(box.conf[0])     # confidence
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw boxes with color by class
        color = (
            (0, 255, 0) if label == "green"
            else (0, 255, 255) if label == "yellow"
            else (0, 0, 255)
        )
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"{label} {conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        # Count
        if label in counts:
            counts[label] += 1

    counts["total"] = sum(counts.values())

    # Save overlay image
    overlay_path = os.path.join(out_dir, f"{base}_overlay.jpg")
    cv2.imwrite(overlay_path, overlay)

    # Save counts to CSV
    csv_path = os.path.join(out_dir, f"{base}_counts.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Green", "Yellow", "Red", "Total"])
        writer.writerow([
            counts["green"],
            counts["yellow"],
            counts["red"],
            counts["total"],
        ])

    return counts, overlay_path


# -----------------------------
# Main Pipeline
# -----------------------------
def main(image_path, out_dir="output", conf=0.15, iou=0.5):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    # Run YOLO detection
    results = model.predict(image_path, conf=conf, iou=iou, verbose=False)[0]

    # Save results
    counts, overlay_path = _save_results(image_path, img, results, out_dir)
    return counts, overlay_path
