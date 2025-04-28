import os
import torch
import torchvision
from torchvision.models import mobilenet_v3_large
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor
import time
from datetime import timedelta
from torchvision.utils import make_grid
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MultiLabelBinarizer
from albumentations import (
    Compose, Resize, HorizontalFlip, Rotate, RandomBrightnessContrast, 
    GaussianBlur, HueSaturationValue, Normalize, MotionBlur, 
    CLAHE, RandomShadow
)
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

model_path = 'Model_to_test.pth'
video_path = "video.avi"
# Define the dictionaries globally
label_to_int = {'Car': 0, 'Bus': 1, 'Motorcycle': 2, 'Truck': 3, 'Ambulance': 4}
int_to_label = {i: label for label, i in label_to_int.items()}

# Number of classes in your dataset
num_classes = 5  # replace with your number of classes :("Car", "Bus", "Motorcycle", "Truck", "Ambulance")

# Number of coordinates for AABB (x_min, y_min, x_max, y_max)
num_coords = 4

num_epochs = 200  # replace with the number of epochs you want to train for

num_boxes = 10  # replace with the number of bounding boxes you want to predict
# Define the weights
class_loss_weight = 1.0
bbox_loss_weight = 1.0 #to miss the bbox coordinates a bit isn't as bad as missing the class or the number of them
num_classes_weight = 1.0
# Define the image width and height after transformation
image_width = 224
image_height = 224


# Transformations with augmentations for training data
train_transform = Compose([
    Resize(image_width, image_width),
    Rotate(limit=5, p=0.5),
    RandomBrightnessContrast(p=0.2),
    GaussianBlur(blur_limit=(3, 5), p=0.2),
    HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2),
    MotionBlur(blur_limit=7, p=0.5),
    CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=0.5),
    RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, p=0.5),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Transformations without augmentations for validation/testing data
valid_test_transform = Compose([
    Resize(image_width, image_height),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Custom collate function to combine images, bounding boxes, labels, and the dictionary
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    bboxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]  # Keep labels as a list of tensors
    num_classes_list = [item[3] for item in batch]  # Add this line
    return images, bboxes, labels, num_classes_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes , num_coords, num_boxes):
        super(ObjectDetectionModel, self).__init__()

        # Load the pretrained MobileNetV3-Large model
        self.backbone = mobilenet_v3_large(pretrained=True)

        # Remove the last layer to use it as a feature extractor
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add a Region Proposal Network (RPN)
        self.rpn = nn.Sequential(
        nn.Conv2d(960, 1024, kernel_size=3, stride=1, padding=1),
        nn.LayerNorm([1024, 1, 1]),
        )
        
        # Add a ROI Pooling layer
        self.roi_pool = nn.AdaptiveMaxPool2d((7, 7))

        num_features = 7 * 7 * 1024  # output size of the ROI Pooling layer when flattened

        # Add a new classifier for class scores
        self.cls_score = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes * num_boxes),  # output layer for class scores
        )

        # Add a new classifier for bounding box coordinates
        self.bbox_pred = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_coords * num_boxes),  # output layer for bounding box coordinates
            nn.Sigmoid()  # Add Sigmoid activation function
        )

        # Add a new classifier for the number of classes
        self.num_classes_pred = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_boxes),  # output layer for the number of classes prediction
        )

    def forward(self, x):
        # Pass the input through the backbone
        x = self.backbone(x)

        # Pass the output through the RPN
        rpn_out = self.rpn(x)

        # Apply ROI Pooling
        roi_out = self.roi_pool(rpn_out)

        # Flatten the output
        roi_out = roi_out.view(roi_out.size(0), -1)

        # Get the class scores and bounding box predictions
        cls_scores = self.cls_score(roi_out)
        bbox_preds = self.bbox_pred(roi_out) * image_width  # Scale the output to [0, 224]
        
        # Get the number of classes prediction
        num_classes_pred = self.num_classes_pred(roi_out)

        # Reshape the outputs to have separate predictions for each bounding box
        cls_scores = cls_scores.view(cls_scores.size(0), -1, num_classes)
        bbox_preds = bbox_preds.view(bbox_preds.size(0), -1, num_coords)

        return cls_scores, bbox_preds, num_classes_pred

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, original_size=(416, 416), new_size=(224, 224)):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform
        self.original_size = original_size
        self.new_size = new_size

    def __len__(self):
        return len(self.annotations['filename'].unique())  # return the number of unique images

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)  # Convert to NumPy array
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR

        # Get all annotations for this image
        image_annotations = self.annotations[self.annotations['filename'] == self.annotations.iloc[index, 0]]

        bboxes = []
        labels = []
        for _, row in image_annotations.iterrows():
            bbox = torch.tensor([float(x) for x in row[4:8]])
            # Adjust the bounding box coordinates
            bbox = self.adjust_bboxes(bbox)
            bboxes.append(bbox)

            # Use the dictionary to convert the labels to integers
            label = torch.tensor(label_to_int[row[3]])
            labels.append(label)

        num_of_classes_index = len(labels) - 1  # Calculate the number of classes minus one to get its index in the predicitions array

        if self.transform:
            augmented = self.transform(image=image)  # Apply transformations
            image = augmented['image']  # Get transformed image

        return image, torch.stack(bboxes), torch.tensor(labels), torch.tensor(num_of_classes_index)  # return a tensor of bounding boxes and labels, the dictionary, and the number of classes

    def adjust_bboxes(self, bbox):
        # Calculate the scale factors
        x_scale = self.new_size[0] / self.original_size[0]
        y_scale = self.new_size[1] / self.original_size[1]

        # Adjust the bounding box coordinates
        xmin, ymin, xmax, ymax = bbox
        adjusted_bbox = torch.tensor([xmin * x_scale, ymin * y_scale, xmax * x_scale, ymax * y_scale])

        return adjusted_bbox
    
def load_model(num_classes,num_coords, num_boxes, predefined=False, model_path=None):
    # Load the object detection model
    model = ObjectDetectionModel(num_classes , num_coords, num_boxes)
    if not predefined:
        # Freeze the parameters of the backbone (we won't be updating these during training)
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        # Load the pretrained model
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        else:    
            model.load_state_dict(torch.load('pretrainedModel.pth'))

    return model.to(device)


def load_datasets():
    # Load the datasets
    train_dataset = CustomDataset(root_dir="C:/Users/User/Desktop/nn_project_version2/sent_to_another_pc/train", annotation_file="C:/Users/User/Desktop/nn_project_version2/sent_to_another_pc/train/_annotations.csv", transform=train_transform)
    valid_dataset = CustomDataset(root_dir="C:/Users/User/Desktop/nn_project_version2/sent_to_another_pc/valid", annotation_file="C:/Users/User/Desktop/nn_project_version2/sent_to_another_pc/valid/_annotations.csv", transform=valid_test_transform)
    test_dataset = CustomDataset(root_dir="C:/Users/User/Desktop/nn_project_version2/sent_to_another_pc/test", annotation_file="C:/Users/User/Desktop/nn_project_version2/sent_to_another_pc/test/_annotations.csv", transform=valid_test_transform)

    return train_dataset, valid_dataset, test_dataset

def create_data_loaders(train_dataset, valid_dataset, test_dataset):
    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)

    return train_loader, valid_loader, test_loader

def define_loss_functions():
    # Define the loss functions
    class_loss = nn.CrossEntropyLoss()
    bbox_loss = nn.SmoothL1Loss()
    num_classes_loss = nn.CrossEntropyLoss()

    return class_loss, bbox_loss, num_classes_loss

def define_optimizer(model):
    # Define the optimizer with weight decay for L2 regularization
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

    return optimizer

def calculate_accuracy(class_preds, labels):
    # Convert class probabilities to class predictions
    class_preds = class_preds[:len(labels)].argmax(dim=1)
    
    # Calculate the number of correct predictions
    correct_preds = (class_preds == labels).float()
    
    # Calculate the accuracy
    accuracy = (correct_preds.sum() / len(labels)).float()
    
    return accuracy.item()

def calculate_num_classes_accuracy(num_classes_preds, num_classes_correct_index):
    # Convert class probabilities to class predictions
    num_classes_preds = num_classes_preds.argmax(dim=1)
    
    # Calculate the number of correct predictions
    correct_preds = (num_classes_preds == num_classes_correct_index).float()
    
    # Calculate the accuracy
    accuracy = correct_preds.sum()
    
    return accuracy.item()

def calculate_iou(pred_boxes, true_boxes):
    # Calculate the intersection coordinates
    x1 = torch.max(pred_boxes[:len(true_boxes), 0], true_boxes[..., 0])
    y1 = torch.max(pred_boxes[:len(true_boxes), 1], true_boxes[..., 1])
    x2 = torch.min(pred_boxes[:len(true_boxes), 2], true_boxes[..., 2])
    y2 = torch.min(pred_boxes[:len(true_boxes), 3], true_boxes[..., 3])
    
    # Calculate the area of intersection
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    # Calculate the area of each bounding box
    pred_boxes_area = (pred_boxes[:len(true_boxes), 2] - pred_boxes[:len(true_boxes), 0]) * (pred_boxes[:len(true_boxes), 3] - pred_boxes[:len(true_boxes), 1])
    true_boxes_area = (true_boxes[..., 2] - true_boxes[..., 0]) * (true_boxes[..., 3] - true_boxes[..., 1])
    
    # Calculate the area of union
    union = pred_boxes_area + true_boxes_area - intersection
    
    # Calculate the IoU
    iou = intersection / union
    
    return iou.mean().item()

def denormalize(image):
    device = image.device  # Get the device of the image tensor
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return image * std + mean

def overlay_boxes_and_labels(image, predicted_bboxes, predicted_labels, num_classes_pred):
    denormalized_image = denormalize(image)
    image_copy = denormalized_image[0].clone().detach().cpu().numpy().transpose(1, 2, 0)
    image_copy = np.clip(image_copy, 0, 1)  # Ensure values are within [0, 1]
    image_copy = (image_copy * 255).astype(np.uint8)  # Rescale to [0, 255] and change the type to uint8
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    num_classes = num_classes_pred.argmax(dim=1).item() + 1   # Get the number of classes 
    for bbox, label in zip(predicted_bboxes[:num_classes], predicted_labels[:num_classes]):
        bbox = tuple(map(int, bbox))
        cv2.rectangle(image_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(image_copy, int_to_label[label.item()], (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    return image_copy

def train_model(model, train_loader, valid_loader, class_loss, bbox_loss, num_classes_loss, optimizer, num_epochs, writer):
    total_start_time = time.time()
    model.train()  # Set the model to training mode

    best_val_loss = float('inf')  # Best validation loss so far
    patience = 50  # Number of epochs to wait for improvement before stopping
    epochs_without_improvement = 0  # Number of epochs without improvement

    for epoch in range(num_epochs):
        for batch_index, (images, bboxes_list, labels_list, num_classes_list) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            total_loss = 0
            total_accuracy = 0
            total_iou = 0
            total_num_classes_accuracy = 0  # Initialize total accuracy for num_classes_preds

            for image_index in range(len(images)):
                single_image = images[image_index].unsqueeze(0)  # Add an extra dimension to match the model input
                single_bboxes = bboxes_list[image_index].to(device)
                single_labels = labels_list[image_index].to(device)

                # Forward pass
                class_preds, bbox_preds, num_classes_preds = model(single_image)
                #bbox_preds = bbox_preds * torch.tensor([image_width, image_height, image_width, image_height]).to(device)

                # Reshape class_preds and the bbox_preds to match the expected input shape
                class_preds = class_preds.view(-1, num_classes)
                bbox_preds = bbox_preds.view(-1, num_coords)

                # Calculate loss
                class_loss_value = class_loss(class_preds[:len(single_labels)], single_labels)
                bbox_loss_value = bbox_loss(bbox_preds[:len(single_bboxes)], single_bboxes)
                num_classes_loss_value = num_classes_loss(num_classes_preds, num_classes_list[image_index].unsqueeze(0).to(device))
                total_loss += class_loss_weight * class_loss_value + bbox_loss_weight * bbox_loss_value + num_classes_weight * num_classes_loss_value

                # Calculate metrics
                #train_accuracy = calculate_accuracy(class_preds, single_labels)
                #total_accuracy += train_accuracy

                # Calculate num_classes accuracy
                #num_classes_accuracy = calculate_num_classes_accuracy(num_classes_preds, num_classes_list[image_index])
                #total_num_classes_accuracy += num_classes_accuracy

                # Calculate IoU
                #iou = calculate_iou(bbox_preds, single_bboxes)
                #total_iou += iou

                if epoch % 30 == 0 and epoch != 0:
                    image_list = []
                    for image_index in range(len(images)):
                        single_image = images[image_index].unsqueeze(0)  # Add an extra dimension to match the model input
                        single_bboxes = bboxes_list[image_index].to(device)
                        single_labels = labels_list[image_index].to(device)

                        # Forward pass
                        class_preds, bbox_preds, num_classes_preds = model(single_image)

                        # Reshape class_preds and the bbox_preds to match the expected input shape
                        class_preds = class_preds.view(-1, num_classes)
                        bbox_preds = bbox_preds.view(-1, num_coords)

                        overlayed_image = overlay_boxes_and_labels(single_image.squeeze(0).cpu(), bbox_preds.detach().cpu(), class_preds.argmax(dim=1).detach().cpu(), num_classes_preds.detach().cpu())

                        # Convert the overlayed image to uint8
                        overlayed_image_uint8 = cv2.convertScaleAbs(overlayed_image)

                        # Convert the overlayed image back to a tensor and permute the dimensions
                        overlayed_image_tensor = torch.from_numpy(overlayed_image_uint8.transpose(2, 0, 1))

                        # Append the overlayed image tensor to the list
                        image_list.append(overlayed_image_tensor)

                    # Convert the list of tensors into a single tensor
                    image_tensor = torch.stack(image_list)

                    # Log the images to TensorBoard
                    writer.add_images('Overlayed Images train', image_tensor, epoch)

            # Log the loss
            writer.add_scalar('Loss/train', total_loss.item(), epoch * len(train_loader) + batch_index)

            # Log more metrics
            #writer.add_scalar('Accuracy/train', total_accuracy / len(images), epoch)

            # Log num_classes accuracy
            #writer.add_scalar('Accuracy/num_classes', total_num_classes_accuracy / len(images), epoch)

            # Log IoU
            #writer.add_scalar('IoU/train', total_iou / len(images), epoch)

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

        # Validate the model after each epoch
        validate_loss = validate_model(model, valid_loader, class_loss, bbox_loss, num_classes_loss, epoch, writer)
        writer.add_scalar('Loss/valid', validate_loss, epoch)
        writer.flush()
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {validate_loss:.4f}')

        # Check if it's the best validation loss so far
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            epochs_without_improvement = 0
            # Save the model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_without_improvement += 1

        # If there's no improvement for a certain number of epochs, stop the training
        if epochs_without_improvement == patience:
            print('Early stopping')
            break

        if(epoch % 100 == 0 and epoch != 0):
            # Save the model
            torch.save(model.state_dict(), 'model.pth')

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    formatted_total_time = str(timedelta(seconds=total_time))
    print(f'Total training time: {formatted_total_time}')

def validate_model(model, valid_loader, class_loss, bbox_loss, num_classes_loss, epoch, writer):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_class_loss = 0
    total_bbox_loss = 0
    total_num_classes_loss = 0
    total_accuracy = 0
    total_iou = 0
    total_num_classes_accuracy = 0

    with torch.no_grad():
        for batch_index, (images, bboxes_list, labels_list, num_classes_list) in enumerate(valid_loader):
            images = images.to(device)
            batch_loss = 0
            batch_class_loss = 0
            batch_bbox_loss = 0
            batch_num_classes_loss = 0

            total_accuracy = 0  # Reset for each batch
            total_iou = 0  # Reset for each batch
            total_num_classes_accuracy = 0  # Reset for each batch

            for image_index in range(len(images)):
                single_image = images[image_index].unsqueeze(0)  # Add an extra dimension to match the model input
                single_bboxes = bboxes_list[image_index].to(device)
                single_labels = labels_list[image_index].to(device)

                # Forward pass
                class_preds, bbox_preds, num_classes_preds = model(single_image)

                # Reshape class_preds and the bbox_preds to match the expected input shape
                class_preds = class_preds.view(-1, num_classes)
                bbox_preds = bbox_preds.view(-1, num_coords)

                # Calculate loss
                class_loss_value = class_loss(class_preds[:len(single_labels)], single_labels)
                bbox_loss_value = bbox_loss(bbox_preds[:len(single_bboxes)], single_bboxes)
                num_classes_loss_value = num_classes_loss(num_classes_preds, num_classes_list[image_index].unsqueeze(0).to(device))
                loss = class_loss_weight * class_loss_value + bbox_loss_weight * bbox_loss_value + num_classes_weight * num_classes_loss_value
                batch_loss += loss.item()
                batch_class_loss += class_loss_value.item()
                batch_bbox_loss += bbox_loss_value.item()
                batch_num_classes_loss += num_classes_loss_value.item()

                # Calculate metrics
                #valid_accuracy = calculate_accuracy(class_preds, single_labels)
                #total_accuracy += valid_accuracy

                # Calculate num_classes accuracy
                #num_classes_accuracy = calculate_num_classes_accuracy(num_classes_preds, num_classes_list[image_index])
                #total_num_classes_accuracy += num_classes_accuracy

                # Calculate IoU
                #iou = calculate_iou(bbox_preds, single_bboxes)
                #total_iou += iou

                if epoch % 30 == 0 and epoch != 0:
                    image_list = []
                    for image_index in range(int(len(images)/8)):
                        single_image = images[image_index].unsqueeze(0)  # Add an extra dimension to match the model input
                        single_bboxes = bboxes_list[image_index].to(device)
                        single_labels = labels_list[image_index].to(device)

                        # Forward pass
                        class_preds, bbox_preds, num_classes_preds = model(single_image)

                        # Reshape class_preds and the bbox_preds to match the expected input shape
                        class_preds = class_preds.view(-1, num_classes)
                        bbox_preds = bbox_preds.view(-1, num_coords)

                        overlayed_image = overlay_boxes_and_labels(single_image.squeeze(0).cpu(), bbox_preds.detach().cpu(), class_preds.argmax(dim=1).detach().cpu(), num_classes_preds.detach().cpu())

                        # Convert the overlayed image to uint8
                        overlayed_image_uint8 = cv2.convertScaleAbs(overlayed_image)

                        # Convert the overlayed image back to a tensor and permute the dimensions
                        overlayed_image_tensor = torch.from_numpy(overlayed_image_uint8.transpose(2, 0, 1))

                        # Append the overlayed image tensor to the list
                        image_list.append(overlayed_image_tensor)

                    # Convert the list of tensors into a single tensor
                    image_tensor = torch.stack(image_list)

                    # Log the images to TensorBoard
                    writer.add_images('Overlayed Images valid', image_tensor, epoch)

            total_loss += batch_loss
            total_class_loss += batch_class_loss
            total_bbox_loss += batch_bbox_loss
            total_num_classes_loss += batch_num_classes_loss

            # Log the loss
            #writer.add_scalar('Loss/valid', batch_loss, epoch * len(valid_loader) + batch_index)
            #writer.add_scalar('Loss/valid_class', batch_class_loss, epoch * len(valid_loader) + batch_index)
            #writer.add_scalar('Loss/valid_bbox', batch_bbox_loss, epoch * len(valid_loader) + batch_index)
            #writer.add_scalar('Loss/valid_num_classes', batch_num_classes_loss, epoch * len(valid_loader) + batch_index)

            # Log the metrics
            #writer.add_scalar('Accuracy/valid', total_accuracy / len(images), epoch)
            #writer.add_scalar('Accuracy/valid_num_classes', total_num_classes_accuracy / len(images), epoch)
            #writer.add_scalar('IoU/valid', total_iou / len(images), epoch)

    model.train()  # Set the model back to training mode for the next epoch

    return total_loss / len(valid_loader)
   
def test_model(model, test_loader, writer):
    # Test the model
    # Store all the true and predicted labels and bounding boxes
    true_labels = []
    pred_labels = []
    true_bboxes = []
    pred_bboxes = []
    true_num_objects = []
    pred_num_objects = []

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_index, (images, bboxes_list, labels_list, num_classes_list) in enumerate(test_loader):
            image_list = []
            # Move data to device
            images = images.to(device)
            for image_index in range(len(images)):
                single_image = images[image_index].unsqueeze(0)  # Add an extra dimension to match the model input
                single_bboxes = bboxes_list[image_index].to(device)
                single_labels = labels_list[image_index].to(device)

                # Forward pass
                class_preds, bbox_preds, num_classes_preds = model(single_image)
                 # Reshape class_preds and the bbox_preds to match the expected input shape
                class_preds = class_preds.view(-1, num_classes)
                bbox_preds = bbox_preds.view(-1, num_coords)

                overlayed_image = overlay_boxes_and_labels(single_image.squeeze(0).cpu(), bbox_preds.detach().cpu(), class_preds.argmax(dim=1).detach().cpu(), num_classes_preds.detach().cpu())

                # Convert the overlayed image to uint8
                overlayed_image_uint8 = cv2.convertScaleAbs(overlayed_image)

                # Convert the overlayed image back to a tensor and permute the dimensions
                overlayed_image_tensor = torch.from_numpy(overlayed_image_uint8.transpose(2, 0, 1))

                # Append the overlayed image tensor to the list
                image_list.append(overlayed_image_tensor)
                true_num_of_objects_in_image = num_classes_list[image_index].item() + 1 
                pred_num_of_objects_in_image = num_classes_preds.argmax(dim=1).item() + 1

                # Store the true and predicted labels, bounding boxes and number of objects
                true_labels.append(single_labels.tolist())
                pred_labels.append(class_preds[:pred_num_of_objects_in_image].argmax(dim=1).tolist())
                true_bboxes.append(single_bboxes.tolist())
                pred_bboxes.append(bbox_preds[:pred_num_of_objects_in_image].tolist())
                true_num_objects.append(true_num_of_objects_in_image)
                pred_num_objects.append(pred_num_of_objects_in_image)

            # Convert the list of tensors into a single tensor
            image_tensor = torch.stack(image_list)

            # Log the images to TensorBoard
            #writer.add_images(f'Overlayed Images test_batch_{batch_index}', image_tensor, batch_index)

   # After the loop, convert the true and predicted labels to binary matrix form
    mlb = MultiLabelBinarizer(classes=np.arange(num_classes))
    true_labels_bin = mlb.fit_transform(true_labels)
    pred_labels_bin = mlb.transform(pred_labels)
    # Calculate the average precision score for each class
    avg_precision_scores = average_precision_score(true_labels_bin, pred_labels_bin, average=None)

    # Calculate the mean average precision (mAP)
    mAP = np.mean(avg_precision_scores)

    # Calculate mean absolute error for number of objects
    mae = mean_absolute_error(true_num_objects, pred_num_objects)

    # Print the total actual vs predicted number of objects
    print(f'Total actual objects: {sum(true_num_objects)}, Total predicted objects: {sum(pred_num_objects)}')

    print(f'mAP: {mAP}, MAE: {mae}')


def process_video(model):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get the video's frame width, height, and frames per second (fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    with torch.no_grad(): 
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to a tensor and pass it through the model
            frame_tensor = valid_test_transform(image=frame)['image']
            frame_tensor = frame_tensor.unsqueeze(0)  # Add a batch dimension
            frame_tensor = frame_tensor.to(device)  # Move the input data to the GPU
            class_preds, bbox_preds, num_classes_pred = model(frame_tensor)

            # Assuming bbox_preds is your tensor of bounding boxes
            scale_factors = torch.tensor([frame_width / frame_tensor.shape[3], 
                                  frame_height / frame_tensor.shape[2], 
                                  frame_width / frame_tensor.shape[3], 
                                  frame_height / frame_tensor.shape[2]], device='cuda:0')

            bbox_preds = bbox_preds * scale_factors

            # Rescale the frame_tensor to the original frame size
            frame_tensor = F.interpolate(frame_tensor, size=(frame_height, frame_width), mode='bilinear', align_corners=False)

            # Reshape class_preds and the bbox_preds to match the expected input shape
            class_preds = class_preds.view(-1, num_classes)
            bbox_preds = bbox_preds.view(-1, num_coords)

            # Draw the bounding boxes and labels on the frame
            frame_with_boxes = overlay_boxes_and_labels(frame_tensor.squeeze(0), bbox_preds, class_preds.argmax(dim=1), num_classes_pred)

            # Write the frame with boxes to the output video
            out.write(frame_with_boxes)

            # Display the frame
            cv2.imshow('Frame', frame_with_boxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()  # Release the VideoWriter
    cv2.destroyAllWindows()

def main():
    print(device)
    writer = SummaryWriter()
    model = load_model(num_classes,num_coords, num_boxes, False)
    train_dataset, valid_dataset, test_dataset = load_datasets()
    train_loader, valid_loader, test_loader = create_data_loaders(train_dataset, valid_dataset, test_dataset)
    class_loss, bbox_loss, num_classes_loss = define_loss_functions()
    optimizer = define_optimizer(model)
    #--uncomment the lines below to train the model--
    #train_model(model, train_loader, valid_loader, class_loss, bbox_loss, num_classes_loss, optimizer, num_epochs, writer)
    #torch.save(model.state_dict(), 'model.pth')
    pretrained_model = load_model(num_classes,num_coords, num_boxes, True, model_path)
    test_model(pretrained_model, test_loader, writer)
    process_video(pretrained_model)
    writer.flush()
    writer.close()
    
if __name__ == "__main__":
    main()