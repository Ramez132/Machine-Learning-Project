import torch
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import imread

# Load pre-trained MobileNetV3 model
model = models.mobilenet_v3_large(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Load the class labels from the text file
def load_class_labels(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        class_labels = {}
        for line in lines[1:]:  # Skip the first line (header)
            parts = line.strip().split('\t')
            class_id = int(parts[0])
            class_name = parts[1]
            class_labels[class_id] = class_name
    return class_labels

# Define preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_labels = load_class_labels('ImagenetClass_labels.txt')

# Function to load and preprocess images from a folder
def load_and_preprocess_images(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = imread(img_path)
            img = transforms.functional.to_pil_image(img)
            images.append(preprocess(img))
            filenames.append(filename)
    return images, filenames

# Perform inference for a single image
def single_image_inference(image, model):
    with torch.no_grad():
        output = model(image.unsqueeze(0))
    return output

# Visualize the images and their predictions
def visualize_images(images, filenames, predictions, class_labels):
    num_images = len(images)
    if num_images == 1:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(images[0].permute(1, 2, 0))
        ax.set_title(f"Prediction: {class_labels[predictions[0]]}")
        ax.axis('off')
    else:
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        for ax, img, filename, prediction in zip(axes, images, filenames, predictions):
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(f"Prediction: {class_labels[prediction]}")
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# Set the folder containing images
folder_path = "C:/Users/khare/OneDrive/Desktop/NN project/image_for_part_one"

# Load and preprocess images
images, filenames = load_and_preprocess_images(folder_path)

# Perform inference for each image
predictions = []
for img in images:
    probabilities = single_image_inference(img, model)
    predictions.append(probabilities.argmax().item())

# Visualize the images and their predictions
visualize_images(images, filenames, predictions, class_labels)
