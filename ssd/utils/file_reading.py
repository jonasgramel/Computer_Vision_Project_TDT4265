import torch
# import pandas as pd
import os
import numpy as np
from PIL import Image, ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.metrics_calculation import iou
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Dataset(torch.utils.data.Dataset):
	"""
    This class with following functions is gathered from Geeks for Geeks: https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/ 
    Accessed: 09-04-2025
    """

	def __init__(
			self, image_dir, label_dir,
			image_size=300,
			num_classes=1, transform=None
	):

		# Read the csv file with image names and labels 
		self.label_list = [filename for filename in sorted(os.listdir(label_dir))]
		# Image and label directories 
		self.image_dir = image_dir 
		self.label_dir = label_dir 
		# Image size 
		self.image_size = image_size 
		# Transformations 
		self.transform = A.Compose([
			A.Resize(300, 300),
			A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			ToTensorV2()
		], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
		# Number of classes 
		self.num_classes = num_classes 
		# Ignore IoU threshold 
		self.ignore_iou_thresh = 0.5

	def __len__(self): 
		return len(self.label_list) 
	
	def __getitem__(self, idx):
		# Getting the label path
		label_path = os.path.join(self.label_dir, self.label_list[idx])

		# Load YOLO-format boxes: x_center, y_center, width, height, class_label
		yolo_boxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()

		# Getting the image path
		img_path = os.path.join(self.image_dir, os.path.splitext(self.label_list[idx])[0] + ".png")
		image = np.array(Image.open(img_path).convert("RGB"))

		# Get image dimensions
		if "rgb" in img_path:
			img_width = 1920
			img_height = 1208
		elif "lidar" in img_path:
			img_width = 1024
			img_height = 128

		# Convert YOLO to Pascal VOC
		boxes = []
		labels = []
		for box in yolo_boxes:
			x_center, y_center, width, height, class_label = box
			# After loading image (before transforms)
			image = np.array(Image.open(img_path).convert("RGB"))

			# Resize box coordinates to match transform size (300x300)
			x_min = (x_center - width / 2)
			y_min = (y_center - height / 2)
			x_max = (x_center + width / 2)
			y_max = (y_center + height / 2)

			#scale_x = 300 / img_width
			#scale_y = 300 / img_height

			#x_min = (x_center - width / 2) * img_width * scale_x
			#y_min = (y_center - height / 2) * img_height * scale_y
			#x_max = (x_center + width / 2) * img_width * scale_x
			#y_max = (y_center + height / 2) * img_height * scale_y

			#x_min = x_min / 300
			#y_min = y_min / 300
			#x_max = x_max / 300
			#y_max = y_max / 300

			boxes.append([x_min, y_min, x_max, y_max])
			#labels.append(int(class_label))  # usually 0 or 1 for binary
			labels.append(int(class_label) + 1)


		# Apply transforms after preparing the boxes
		if self.transform:
			transformed = self.transform(
				image=image,
				bboxes=boxes,
				class_labels=labels
			)
			image = transformed['image']
			boxes = transformed['bboxes']
			labels = transformed['class_labels']
		# Convert normalized VOC boxes to absolute pixel coords (300x300)
		boxes = np.array(boxes)  # shape: (N, 4)
		boxes[:, [0, 2]] *= self.image_size  # x_min, x_max
		boxes[:, [1, 3]] *= self.image_size  # y_min, y_max

		# Ensure non-empty, well-formed tensors
		if len(boxes) == 0:
			boxes = torch.zeros((0, 4), dtype=torch.float32)
			labels = torch.zeros((0,), dtype=torch.int64)
		else:
			boxes = torch.tensor(boxes, dtype=torch.float32)
			labels = torch.tensor(labels, dtype=torch.int64)

		target = {
			'boxes': boxes,
			'labels': labels
		}

		return image, target




# Create a dataset class to load the images and labels from the folder 
class decomissioned_Dataset(torch.utils.data.Dataset): 
	"""
    This class with following functions is gathered from Geeks for Geeks: https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/ 
    Accessed: 09-04-2025
    """
	def __init__(
		self, image_dir, label_dir, anchors, 
		image_size=300, grid_sizes=[13, 26, 52], 
		num_classes=1, transform=None
	): 
		# Read the csv file with image names and labels 
		self.label_list = [filename for filename in sorted(os.listdir(label_dir))]
		# Image and label directories 
		self.image_dir = image_dir 
		self.label_dir = label_dir 
		# Image size 
		self.image_size = image_size 
		# Transformations 
		self.transform = transform 
		# Grid sizes for each scale 
		self.grid_sizes = grid_sizes 
		# Anchor boxes 
		self.anchors = torch.tensor( 
			anchors[0] + anchors[1] + anchors[2]) 
		# Number of anchor boxes 
		self.num_anchors = self.anchors.shape[0] 
		# Number of anchor boxes per scale 
		self.num_anchors_per_scale = self.num_anchors // 3
		# Number of classes 
		self.num_classes = num_classes 
		# Ignore IoU threshold 
		self.ignore_iou_thresh = 0.5

	def __len__(self): 
		return len(self.label_list) 
	
	def __getitem__(self, idx): 
		# Getting the label path 
		label_path = os.path.join(self.label_dir, self.label_list[idx]) 
		# We are applying roll to move class label to the last column 
		# 5 columns: x, y, width, height, class_label 
		bboxes = np.roll(np.loadtxt(fname=label_path, 
						delimiter=" ", ndmin=2), 4, axis=1).tolist() 
		
		# Getting the image path 
		img_path = os.path.join(self.image_dir, os.path.splitext(self.label_list[idx])[0] + ".png") 
		image = np.array(Image.open(img_path).convert("RGB")) 

		# Albumentations augmentations 
		if self.transform: 
			augs = self.transform(image=image, bboxes=bboxes) 
			image = augs["image"] 
			bboxes = augs["bboxes"] 

		# Below assumes 3 scale predictions (as paper) and same num of anchors per scale 
		# target : [probabilities, x, y, width, height, class_label] 
		targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6)) 
				for s in self.grid_sizes]
		
		# Identify anchor box and cell for each bounding box 
		for box in bboxes: 
			# Calculate iou of bounding box with anchor boxes 
			iou_anchors = iou(torch.tensor(box[2:4]), 
							self.anchors, 
							is_pred=False) 
			# Selecting the best anchor box 
			anchor_indices = iou_anchors.argsort(descending=True, dim=0) 
			x, y, width, height, class_label = box

			# At each scale, assigning the bounding box to the 
			# best matching anchor box 
			has_anchor = [False] * 3
			for anchor_idx in anchor_indices: 
				scale_idx = anchor_idx // self.num_anchors_per_scale 
				anchor_on_scale = anchor_idx % self.num_anchors_per_scale 
				
				# Identifying the grid size for the scale 
				s = self.grid_sizes[scale_idx] 
				
				# Identifying the cell to which the bounding box belongs 
				i, j = int(s * y), int(s * x) 
				anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] 
				
				# Check if the anchor box is already assigned 
				if not anchor_taken and not has_anchor[scale_idx]: 

					# Set the probability to 1 
					targets[scale_idx][anchor_on_scale, i, j, 0] = 1

					# Calculating the center of the bounding box relative 
					# to the cell 
					x_cell, y_cell = s * x - j, s * y - i 

					# Calculating the width and height of the bounding box 
					# relative to the cell 
					width_cell, height_cell = (width * s, height * s) 

					# Idnetify the box coordinates 
					box_coordinates = torch.tensor( 
										[x_cell, y_cell, width_cell, 
										height_cell] 
									) 

					# Assigning the box coordinates to the target 
					targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates 

					# Assigning the class label to the target 
					targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) 

					# Set the anchor box as assigned for the scale 
					has_anchor[scale_idx] = True

				# If the anchor box is already assigned, check if the 
				# IoU is greater than the threshold 
				elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh: 
					# Set the probability to -1 to ignore the anchor box 
					targets[scale_idx][anchor_on_scale, i, j, 0] = -1

		# Return the image and the target 
		return image, tuple(targets)