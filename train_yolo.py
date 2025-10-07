from ultralytics import YOLO

def train_model():
    # Load a pre-trained YOLOv8 nano model (lightweight for laptops)
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data="dataset/train/Coffee Fruit Maturity ---.v1i.yolov8/data.yaml",  
        epochs=20,         # Reduced from 50 to 20 for faster training
        imgsz=640,
        batch=4,           # Smaller batch size to reduce CPU usage and heating
        device="cpu",      # 'cpu' = CPU only
        save=True          # Ensures checkpoints are saved for resuming
    )

    # Print summary
    print("\n✅ Training complete!")
    print(f"📂 Best weights saved at: {results.save_dir}/weights/best.pt")

    # Print dataset info
    print(f"📊 Number of classes: {model.model.nc}")
    print(f"🏷 Class names: {model.model.names}")

if __name__ == "__main__":
    train_model()