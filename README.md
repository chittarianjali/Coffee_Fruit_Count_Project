# Coffee Fruit Counter (Color-Space Weighting + Ellipse Counting)

A simple, classical-computer-vision pipeline inspired by *"Estimation of fruit number in coffee trees by maturity level, based on color space weighting, using a new segmentation algorithm"*.  
This implementation uses **color-space weighting** across HSV, Lab, and RGB to segment **green / yellow / red** coffee cherries, followed by morphological cleanup and **ellipse-based instance counting**.

> ⚠️ This is a lightweight reference implementation for quick experimentation. You will likely tune thresholds per camera/lighting and orchard.

## Features
- Multi-space color weighting (HSV + Lab + RGB) for three maturity masks.
- Background suppression and morphology to reduce leaves/branches.
- Ellipse fitting per connected component + size & circularity filters.
- Batch CLI: process a folder of images, export overlays and a CSV with counts per image and per maturity level.

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python -m src.pipeline --input path/to/images --out out_dir
# Or for a single image
python -m src.pipeline --input path/to/image.jpg --out out_dir
```

### Important knobs
- `--min-area`, `--max-area`: pixel area range for a cherry in your images.
- `--debug`: writes intermediate masks to the output directory.
- `--show`: display windows (useful locally).

### Outputs
- `out/overlays/<image>_overlay.png`: colored contours per class.
- `out/results.csv`: rows per image with counts: total, green, yellow, red.

## Notes
- Works best with images where cherries are reasonably in focus and not too motion-blurred.
- If your cherries are small or large in pixels, adjust the area range.
- You may tweak color thresholds in `segmentation.py` for your camera/lighting.
