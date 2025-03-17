import pandas as pd
import os

# Define label mapping with class ID
label_map = {
    "Front-Windscreen-Damage": 0,
    "Headlight-Damage": 1,
    "Major-Rear-Bumper-Dent": 2,
    "Rear-windscreen-Damage": 3,
    "RunningBoard-Dent": 4,
    "Sidemirror-Damage": 5,
    "Signlight-Damage": 6,
    "Taillight-Damage": 7,
    "bonnet-dent": 8,
    "doorouter-dent": 9,
    "fender-dent": 10,
    "front-bumper-dent": 11,
    "medium-Bodypanel-Dent": 12,
    "pillar-dent": 13,
    "quaterpanel-dent": 14,
    "rear-bumper-dent": 15,
    "roof-dent": 16
}

# Read CSV file
df = pd.read_csv("_classes.csv")

# Output directory for YOLO labels
output_dir = "labels_yolo"
os.makedirs(output_dir, exist_ok=True)

# Generate dummy bbox coordinates (since CSV does not have bbox, need to fix if there are real bbox)
def generate_dummy_bbox():
    return 0.5, 0.5, 0.4, 0.4  # x_center, y_center, width, height

# Convert each image
for _, row in df.iterrows():
    filename = row.iloc[0]  # Image file name
    label_txt = []
    
    for label, value in row.iloc[1:].items():
        if value == 1:  # If the image has this defect
            class_id = label_map[label]
            x, y, w, h = generate_dummy_bbox()
            label_txt.append(f"{class_id} {x} {y} {w} {h}")
    
    # Save txt file
    if label_txt:
        with open(os.path.join(output_dir, f"{filename.split('.')[0]}.txt"), "w") as f:
            f.write("\n".join(label_txt))

print("✅ Conversion complete! YOLO labels saved in directory:", output_dir)
