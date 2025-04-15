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

from utils.box_preparation import convert_cells_to_bboxes
from utils.file_reading import Dataset
from utils.evaluation import nms, YOLOLoss
from model_structure import YOLOv3
from utils.metrics_calculation import mean_average_precision

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
learning_rate = 1e-5

# Number of epochs for training 
epochs = 10

# Image size 
image_size = 416

# Grid cell sizes 
s = [image_size // 32, image_size // 16, image_size // 8] 

# Class labels 

class_labels = ["Pole"]

# Function to plot images with bounding boxes and class labels 
def plot_image(image, boxes, image_index): 
    """
    This function is gathered from Geeks for Geeks: https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/ 
    Accessed: 09-04-2025
    """
    # Getting the color map from matplotlib 
    colour_map = plt.get_cmap("tab20b") 
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))] 
  
    # Reading the image with OpenCV 
    img = np.array(image) 
    # Getting the height and width of the image 
    h, w, _ = img.shape 
  
    # Create figure and axes 
    fig, ax = plt.subplots(1) 
  
    # Add image to plot 
    ax.imshow(img) 
    # print("Number of boxes: ", len(boxes))
    # Plotting the bounding boxes and labels over the image 
    for box in boxes: 
        # print("Box: ", box)
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
    plt.savefig("figures/figure"+str(image_index))
    
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
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True) 
    model.load_state_dict(checkpoint["state_dict"]) 
    optimizer.load_state_dict(checkpoint["optimizer"]) 
  
    for param_group in optimizer.param_groups: 
        param_group["lr"] = lr 

# Transform for training 
train_transform = A.Compose( 
	[ 
		# Rescale an image so that maximum side is equal to image_size 
		A.LongestMaxSize(max_size=image_size), 
		# Pad remaining areas with zeros 
		A.PadIfNeeded( 
			min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
		), 
		# Random color jittering 
		A.ColorJitter( 
			brightness=0.5, contrast=0.5, 
			saturation=0.5, hue=0.5, p=0.5
		), 
		# Flip the image horizontally 
		A.HorizontalFlip(p=0.5), 
		# Normalize the image 
		A.Normalize( 
			mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
		), 
		# Convert the image to PyTorch tensor 
		ToTensorV2() 
	], 
	# Augmentation for bounding boxes 
	bbox_params=A.BboxParams( 
					format="yolo", 
					min_visibility=0.4, 
					label_fields=[] 
				) 
) 

# Transform for testing 
test_transform = A.Compose( 
	[ 
		# Rescale an image so that maximum side is equal to image_size 
		A.LongestMaxSize(max_size=image_size), 
		# Pad remaining areas with zeros 
		A.PadIfNeeded( 
			min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
		), 
		# Normalize the image 
		A.Normalize( 
			mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
		), 
		# Convert the image to PyTorch tensor 
		ToTensorV2() 
	], 
	# Augmentation for bounding boxes 
	bbox_params=A.BboxParams( 
					format="yolo", 
					min_visibility=0.4, 
					label_fields=[] 
				) 
)


# Creating a dataset object 
dataset = Dataset( 
	#csv_file="train.csv", 
	image_dir="/work/datasets/tdt4265/ad/open/Poles/lidar/combined_color/train", 
	label_dir="/work/datasets/tdt4265/ad/open/Poles/lidar/labels/train", 
	
	
    grid_sizes=[13, 26, 52], 
	anchors=ANCHORS, 
	transform=test_transform 
) 

# Creating a dataloader object 
loader = torch.utils.data.DataLoader( 
	dataset=dataset, 
	batch_size=batch_size, 
	shuffle=True, 
) 

# Defining the grid size and the scaled anchors 
GRID_SIZE = [13, 26, 52] 
scaled_anchors = torch.tensor(ANCHORS) / ( 
	1 / torch.tensor(GRID_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
) 

# Getting a batch from the dataloader 
x, y = next(iter(loader)) 

# Getting the boxes coordinates from the labels 
# and converting them into bounding boxes without scaling 
boxes = [] 
for i in range(y[0].shape[1]): 
	anchor = scaled_anchors[i] 
	boxes += convert_cells_to_bboxes( 
			y[i], is_predictions=False, s=y[i].shape[2], anchors=anchor 
			)[0] 

# Applying non-maximum suppression 
boxes = nms(boxes, iou_threshold=1, threshold=0.7) 

# Plotting the image with the bounding boxes 
plot_image(x[0].permute(1,2,0).to("cpu"), boxes,-1)


# Define the train function to train the model 
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors): 
    # Creating a progress bar 
    progress_bar = tqdm(loader, leave=True) 

    # Initializing a list to store the losses 
    losses = [] 

    # Iterating over the training data 
    for _, (x, y) in enumerate(progress_bar): 
        x = x.to(device) 
        y0, y1, y2 = ( 
            y[0].to(device), 
            y[1].to(device), 
            y[2].to(device), 
        ) 

        with torch.amp.autocast('cuda'): 
            # Getting the model predictions 
            outputs = model(x) 
            # Calculating the loss at each scale 
            loss = ( 
                loss_fn(outputs[0], y0, scaled_anchors[0]) 
                + loss_fn(outputs[1], y1, scaled_anchors[1]) 
                + loss_fn(outputs[2], y2, scaled_anchors[2]) 
            ) 

        # Add the loss to the list 
        losses.append(loss.item()) 

        # Reset gradients 
        optimizer.zero_grad() 

        # Backpropagate the loss 
        scaler.scale(loss).backward() 

        # Optimization step 
        scaler.step(optimizer) 
        
        # Update the scaler for next iteration 
        scaler.update() 
        # update progress bar with loss 
        mean_loss = sum(losses) / len(losses) 
        progress_bar.set_postfix(loss=mean_loss)
    
    return losses, mean_loss   

trainyolov3 = False
test_yolov3 = True


if __name__ == "__main__": 

	if trainyolov3:
        # Creating the model from YOLOv3 class 
		model = YOLOv3().to(device) 

		# Defining the optimizer 
		optimizer = optim.Adam(model.parameters(), lr = learning_rate) 

		# Defining the loss function 
		loss_fn = YOLOLoss() 

		# Defining the scaler for mixed precision training 
		scaler = torch.cuda.amp.GradScaler() 

		# Defining the train dataset 
		train_dataset = Dataset( 
			# csv_file="./data/pascal voc/train.csv", 
			image_dir = "/work/datasets/tdt4265/ad/open/Poles/lidar/combined_color/train", # For Cybele, lidar images
			label_dir = "/work/datasets/tdt4265/ad/open/Poles/lidar/labels/train", # For Cybele, lidar labels
			anchors=ANCHORS, 
			transform=train_transform 
		) 

		# Defining the train data loader 
		train_loader = torch.utils.data.DataLoader( 
			train_dataset, 
			batch_size = batch_size, 
			num_workers = 2, 
			shuffle = True, 
			pin_memory = True, 
		) 

		# Scaling the anchors 
		scaled_anchors = ( 
			torch.tensor(ANCHORS) *
			torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2) 
		).to(device) 
		losses_list = []
		plt.figure()

		# Training the model 
		for e in range(1, epochs+1): 
			print("Epoch:", e) 
			losses, mean_loss = training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)			
			losses_list.extend(losses)

			# Saving the model
		if save_model:
			save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

		plt.plot(losses_list)
		plt.savefig("figures/error_plot")

	if test_yolov3:
        # Setting number of classes and image size 
		num_classes = 1
		IMAGE_SIZE = 416

		# Creating model and testing output shapes 
		model = YOLOv3(num_classes=num_classes) 
		x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE)) 
		out = model(x) 
		print(out[0].shape) 
		print(out[1].shape) 
		print(out[2].shape) 
	
		# Asserting output shapes 
		assert model(x)[0].shape == (1, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5) 
		assert model(x)[1].shape == (1, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5) 
		assert model(x)[2].shape == (1, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5) 
		print("Output shapes are correct!")

		# Taking a sample image and testing the model 
	
		# Setting the load_model to True 
		load_model = True
		
		# Defining the model, optimizer, loss function and scaler 
		model = YOLOv3().to(device) 
		optimizer = optim.Adam(model.parameters(), lr = learning_rate) 
		loss_fn = YOLOLoss() 
		scaler = torch.amp.GradScaler('cuda') 
		
		# Loading the checkpoint 
		if load_model: 
			load_checkpoint(checkpoint_file, model, optimizer, learning_rate ) 
		
		# Defining the test dataset and data loader 
		val_dataset = Dataset( 
			# csv_file="./data/pascal voc/test.csv", 
			image_dir = "/work/datasets/tdt4265/ad/open/Poles/lidar/combined_color/valid", # For Cybele, lidar images
			# label_dir="./data/pascal voc/labels/", 
			label_dir = "/work/datasets/tdt4265/ad/open/Poles/lidar/labels/valid", # For Cybele, lidar labels
			anchors=ANCHORS, 
			transform=test_transform
			# label_dir="./data/pascal voc/labels/",  
		) 
		val_loader = torch.utils.data.DataLoader( 
			val_dataset, 
			batch_size = batch_size, 
			num_workers = 2, 
			shuffle = True, 
		)
		
		# Getting a sample image from the test data loader 
		# x, y = next(iter(val_loader)) 
		# x = x.to(device) 
		
		all_predictions = []
		all_gt_boxes = []
		model.eval() 

		with torch.no_grad(): 
			# Getting the model predictions 
			for x, y in val_loader:
				x = x.to(device) 
				batch_size = x.shape[0]
				# Getting the model predictions 
				output = model(x) 
				# Getting the bounding boxes from the predictions 
				bboxes = [[] for _ in range(x.shape[0])] 
				anchors = ( 
						torch.tensor(ANCHORS) 
							* torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) 
						).to(device) 
			
				# Getting bounding boxes for each scale 
				for i in range(3): 
					batch_size, A, S, _, _ = output[i].shape 
					anchor = anchors[i] 
					boxes_scale_i = convert_cells_to_bboxes( 
										output[i], anchor, s=S, is_predictions=True
									) 
					for idx, (box) in enumerate(boxes_scale_i): 
						bboxes[idx] += box 
				for i in range(3):
					batch_size, A, S, _, _ = y[i].shape 
					anchor = anchors[i] 
					gt_boxes = convert_cells_to_bboxes(y[i], anchors, s=S, is_predictions=False)
					all_gt_boxes.append(gt_boxes)
			# Plotting the image with bounding boxes for each image in the batch 
				for i in range(0,batch_size,10): 
					# Applying non-max suppression to remove overlapping bounding boxes 
					nms_boxes = nms(bboxes[i], iou_threshold=0.5, threshold=0.6) 
					all_predictions.append(nms_boxes)
					# Plotting the image with bounding boxes 
					plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes, i)
		print("GT boxes: ", all_gt_boxes)
		print("Predictions: ", all_predictions)
		# Calculating mean average precision
		precisions, recall, mean_average_precision = mean_average_precision(all_gt_boxes, all_predictions)