from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Perform prediction on an example image
results = model.predict("./dataset/test/test_3.jpg", save=True)
