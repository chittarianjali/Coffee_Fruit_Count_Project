import argparse
import os
from src.pipeline import main

def get_latest_uploaded(path="static/uploads"):
    """Return the most recent file from static/uploads."""
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if not files:
        raise FileNotFoundError(f"No files found in {path}")
    return max(files, key=os.path.getctime)  # latest by creation time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--min-area", type=int, default=100, help="Minimum contour area")
    parser.add_argument("--max-area", type=int, default=8000, help="Maximum contour area")
    args = parser.parse_args()

    # If placeholder image is given, replace with the latest uploaded file
    if "your_image.jpg" in args.input:
        print("INFO: 'your_image.jpg' detected, fetching latest upload instead...")
        args.input = get_latest_uploaded("static/uploads")
        print("Using:", args.input)

    counts, images = main(args.input, args.out, args.min_area, args.max_area)
    print("Counts:", counts)
    print("Images:", images)
