from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Define the preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size of the model
        transforms.ToTensor(),           # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])
    
    # Apply the transformations
    image_tensor = preprocess(image)
    
    return image_tensor.unsqueeze(0)  # Add batch dimension

def load_and_preprocess_image(image_path):
    return preprocess_image(image_path)