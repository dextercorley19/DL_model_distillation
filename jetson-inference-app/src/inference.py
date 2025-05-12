import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import sys

# Import the class labels
from class_labels import class_labels

# Define the path to the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/checkpoint_14.pth')

# Define the model architecture
def get_model(num_classes):
    model = models.resnet18(pretrained=False)  # Use ResNet-18
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Adjust the final layer
    return model

# Load the trained model
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(class_labels)  # Number of classes
    model = get_model(num_classes)  # Initialize the model architecture
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # Load the state dictionary
    model.eval()  # Set the model to evaluation mode
    return model.to(device)

# Preprocess the input image
def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size of the model
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Run inference on the input image
def run_inference(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Main function to execute inference
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    image_tensor = preprocess_image(image_path, device)  # Pass the device to preprocess_image
    prediction = run_inference(model, image_tensor)

    print(f"Predicted class: {class_labels[prediction]}")