import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

# Load model
class CarDamageModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(CarDamageModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 56 * 56, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load label mapping
label_map = {"scratch": 0, "seat_damage": 1, "tire_worn": 2, "dent": 3}
num_classes = len(label_map)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CarDamageModel(num_classes).to(device)
model.load_state_dict(torch.load("car_damage_model.pth"))
model.eval()

# Transform for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Function to predict a single image
def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    label = list(label_map.keys())[list(label_map.values()).index(predicted.item())]
    
    # Dummy bounding box (Placeholder for actual object detection model)
    bbox = {"x": 0.3, "y": 0.4, "width": 0.4, "height": 0.5}
    
    return {
        "label": label,
        "confidence": round(confidence.item(), 2),
        **bbox
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    result = predict_image(image)
    return {"filename": file.filename, "predictions": [result]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
