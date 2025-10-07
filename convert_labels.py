import os

# Base dataset path
base_path = "dataset/train/Coffee Fruit Maturity ---.v1i.yolov8"

# Dataset splits
splits = ["train", "valid", "test"]

# Mapping: old class → new class
class_map = {
    0: 2,  # dry → red
    1: 2,  # overripe → red
    2: 2,  # ripe → red
    3: 1,  # semi_ripe → yellow
    4: 0   # unripe → green
}

for split in splits:
    orig_labels_path = os.path.join(base_path, split, "labels")
    new_labels_path = os.path.join(base_path, split, "labels_3class")
    os.makedirs(new_labels_path, exist_ok=True)

    for file_name in os.listdir(orig_labels_path):
        if file_name.endswith(".txt"):
            old_path = os.path.join(orig_labels_path, file_name)
            new_path = os.path.join(new_labels_path, file_name)

            with open(old_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    old_class = int(parts[0])
                    if old_class in class_map:
                        new_class = class_map[old_class]
                        parts[0] = str(new_class)
                        new_lines.append(" ".join(parts))

            with open(new_path, "w") as f:
                f.write("\n".join(new_lines))

    print(f"✅ Converted {split} labels → saved in {new_labels_path}")
