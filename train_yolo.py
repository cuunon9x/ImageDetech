from ultralytics import YOLO

# Load YOLOv8 model (small model for quick training, can be replaced with yolov8m.pt, yolov8l.pt, etc.)
model = YOLO("yolov8n.pt")

# Train model
model.train(
    data="./data.yaml",  # Dataset configuration file
    epochs=50,         # Number of epochs (can be increased if needed)
    imgsz=640,         # Image size
    batch=16,          # Batch size
    device="cpu"      # Use GPU (if available)
)
