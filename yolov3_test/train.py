import torch 
import torch.nn as nn 
import torch.optim as optim 

from PIL import Image, ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True

import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import cv2 

import os 
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import matplotlib.patches as patches 

from tqdm import tqdm


### Defining paramters and objects:

# Device 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and save model variable 
load_model = False
save_model = True

# model checkpoint file name 
checkpoint_file = "checkpoint.pth.tar"

# Anchor boxes for each feature map scaled between 0 and 1 
# 3 feature maps at 3 different scales based on YOLOv3 paper 
ANCHORS = [ 
	[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
	[(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
	[(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
] 

# Batch size for training 
batch_size = 32

# Learning rate for training 
leanring_rate = 1e-5

# Number of epochs for training 
epochs = 20

# Image size 
image_size = 416

# Grid cell sizes 
s = [image_size // 32, image_size // 16, image_size // 8] 

# Class labels 
class_labels = [ 
	"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
	"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
	"pottedplant", "sheep", "sofa", "train", "tvmonitor"
]



# Function to plot images with bounding boxes and class labels 
def plot_image(image, boxes): 
    """
    This function is gathered from Geeks for Geeks: https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/ 
    Accessed: 09-04-2025
    """
    # Getting the color map from matplotlib 
    colour_map = plt.get_cmap("tab20b") 
    # Getting 20 different colors from the color map for 20 different classes 
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))] 
  
    # Reading the image with OpenCV 
    img = np.array(image) 
    # Getting the height and width of the image 
    h, w, _ = img.shape 
  
    # Create figure and axes 
    fig, ax = plt.subplots(1) 
  
    # Add image to plot 
    ax.imshow(img) 
  
    # Plotting the bounding boxes and labels over the image 
    for box in boxes: 
        # Get the class from the box 
        class_pred = box[0] 
        # Get the center x and y coordinates 
        box = box[2:] 
        # Get the upper left corner coordinates 
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
  
        # Create a Rectangle patch with the bounding box 
        rect = patches.Rectangle( 
            (upper_left_x * w, upper_left_y * h), 
            box[2] * w, 
            box[3] * h, 
            linewidth=2, 
            edgecolor=colors[int(class_pred)], 
            facecolor="none", 
        ) 
          
        # Add the patch to the Axes 
        ax.add_patch(rect) 
          
        # Add class name to the patch 
        plt.text( 
            upper_left_x * w, 
            upper_left_y * h, 
            s=class_labels[int(class_pred)], 
            color="white", 
            verticalalignment="top", 
            bbox={"color": colors[int(class_pred)], "pad": 0}, 
        ) 
  
    # Display the plot 
    plt.show()
    
# Function to save checkpoint 
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"): 
    """
    This function is gathered from Geeks for Geeks: https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/ 
    Accessed: 09-04-2025
    """
    print("==> Saving checkpoint") 
    checkpoint = { 
		"state_dict": model.state_dict(), 
		"optimizer": optimizer.state_dict(), 
	} 
    torch.save(checkpoint, filename)

# Function to load checkpoint 
def load_checkpoint(checkpoint_file, model, optimizer, lr): 
    """
    This function is gathered from Geeks for Geeks: https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/ 
    Accessed: 09-04-2025
    """
    print("==> Loading checkpoint") 
    checkpoint = torch.load(checkpoint_file, map_location=device) 
    model.load_state_dict(checkpoint["state_dict"]) 
    optimizer.load_state_dict(checkpoint["optimizer"]) 
  
    for param_group in optimizer.param_groups: 
        param_group["lr"] = lr 

