from flask import Flask, render_template, request
from ultralytics import YOLO
import os, cv2

app = Flask(__name__)
model = YOLO("runs/detect/train8/weights/best.pt")  # ✅ your trained weights

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        # Save uploaded image
        uploaded_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(uploaded_path)

        # Run YOLOv8 inference
        results = model(uploaded_path)

        # Initialize counters with all classes
        counts = {"green": 0, "yellow": 0, "red": 0}

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                label = model.names[cls_id]

                if label in counts:   # only count relevant classes
                    counts[label] += 1

        total = sum(counts.values())

        # Save analysed image
        output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
        cv2.imwrite(output_path, results[0].plot())

        return render_template(
            "result.html",
            uploaded=file.filename,   # pass uploaded image filename
            image="output.jpg",       # analysed image
            counts=counts,
            total=total
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
