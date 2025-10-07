from ultralytics import YOLO
import cv2
import os

# Load trained model
model = YOLO(r"runs/detect/train8/weights/best.pt")

# Path to images (can point to training folder or any folder with images)
img_folder = r"dataset/train/Coffee Fruit Maturity ---.v1i.yolov8/train/images"

# Get all image files
images = [os.path.join(img_folder, f) for f in os.listdir(img_folder)
          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Class mapping
class_names = {0: 'green', 1: 'yellow', 2: 'red'}

# Output folder
os.makedirs("output", exist_ok=True)

for img_path in images:
    results = model(img_path)
    result = results[0]

    counts = {'green': 0, 'yellow': 0, 'red': 0}
    for cls_id in result.boxes.cls:
        counts[class_names[int(cls_id)]] += 1
    total = sum(counts.values())

    print(f"\nImage: {os.path.basename(img_path)}")
    print(f"Green: {counts['green']}, Yellow: {counts['yellow']}, Red: {counts['red']}, Total: {total}")

    # Save annotated image
    annotated_img = result.plot()
    save_path = os.path.join("output", os.path.basename(img_path))
    cv2.imwrite(save_path, annotated_img)
