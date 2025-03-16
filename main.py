import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Define data directory
data_dir = "./dataset"
labels_file = "labels.csv"
img_size = 224  # Standard image size
batch_size = 16
epochs = 10  # Number of epochs to train

# Read label file
df = pd.read_csv(labels_file)
print(df.head())  # Check input data

# Split train, val, test sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'])

print(train_df.shape, val_df.shape, test_df.shape)  # Check sample sizes

# Convert labels to numbers
label_map = {label: i for i, label in enumerate(df['label'].unique())}
print(label_map)  # Check label mapping

train_df['label'] = train_df['label'].map(label_map)
val_df['label'] = val_df['label'].map(label_map)
test_df['label'] = test_df['label'].map(label_map)

# Define dataset class
class CarDamageDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        folder_name = '_'.join(img_name.split('_')[:-1])  # Extract folder name from image name
        img_path = os.path.join(self.data_dir, folder_name, img_name)
        if not os.path.exists(img_path):
            print(f"Image file {img_path} not found.")
            return None
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read image file {img_path}.")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.dataframe.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# Prepare transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Create DataLoader
train_dataset = CarDamageDataset(train_df, data_dir, transform)
val_dataset = CarDamageDataset(val_df, data_dir, transform)
test_dataset = CarDamageDataset(test_df, data_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Preprocessing complete! Data has been split into train, val, test.")

# Define a simple AI model
class CarDamageModel(nn.Module):
    def __init__(self, num_classes):
        super(CarDamageModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
num_classes = len(label_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CarDamageModel(num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to train model
def train_model(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
        
        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss}, Accuracy: {accuracy}%")

# Train model
train_model(model, train_loader, val_loader, epochs)
print("Training complete!")

# Save model
torch.save(model.state_dict(), "car_damage_model.pth")
print("Model saved!")

# Function to predict a single image
def predict_image(image_path, model, transform):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    label = list(label_map.keys())[list(label_map.values()).index(predicted.item())]
    print(f"Image {image_path} is predicted as: {label}")
    return label

# Test model with a real image
image_path = "./dataset/test/test_1.jpg"  # Replace with your image path
predict_image(image_path, model, transform)

# Function to evaluate model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

# Evaluate model
evaluate_model(model, test_loader)
