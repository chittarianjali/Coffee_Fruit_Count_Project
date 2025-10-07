from ultralytics.data.utils import check_det_dataset
from ultralytics import YOLO

# Check dataset structure
dataset_info = check_det_dataset("dataset/train/Coffee Fruit Maturity ---.v1i.yolov8/data.yaml")
print(dataset_info)

# Quick visualization run (1 epoch)
model = YOLO("yolov8n.pt")
model.train(
    data="dataset/train/Coffee Fruit Maturity ---.v1i.yolov8/data.yaml",
    epochs=1,
    imgsz=416,
    device="cpu"
)
