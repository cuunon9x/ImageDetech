from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import torch
import io
import uvicorn

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Run YOLO model
    results = model.predict(image)
    detections = []
    
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            detections.append({
                "label": model.names[int(cls)],
                "confidence": round(conf, 2),
                "x": round((x1 + x2) / 2, 4),
                "y": round((y1 + y2) / 2, 4),
                "width": round(x2 - x1, 4),
                "height": round(y2 - y1, 4)
            })
    
    return {"filename": file.filename, "predictions": detections}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
