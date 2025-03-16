import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

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
model.load_state_dict(torch.load("./car_damage_model.pth"))
model.eval()

# Transform for images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Function to predict a single image
def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    label = list(label_map.keys())[list(label_map.values()).index(predicted.item())]
    print(f"Image {image_path} is predicted as: {label}")
    return label

if __name__ == "__main__":
    image_path = "./dataset/test/test_2.jpg"  # Replace with your image path
    predict_image(image_path)
