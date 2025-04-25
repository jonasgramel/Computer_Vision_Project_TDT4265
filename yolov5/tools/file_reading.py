import torch
# import pandas as pd
import os
import numpy as np
from PIL import Image, ImageFile
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.ops import box_iou

def get_bounding_boxes(label_file):
    """
    Based on Listing 1 in Bavirisetti et al. (2025): https://www.sciencedirect.com/science/article/pii/S2352340925001350?via%3Dihub
    Load bounding boxes from the label file.

    Input:
        - label_file (str): Path to the label file
    Output:
        - bboxes (numpy array): Array of bounding boxes in YOLO format
    """
    bboxes = []
    with open(label_file, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        try:
            parts = list(map(float, line.strip().split()))
            if len(parts) == 5:   # Ensure correct YOLO format: class_id x_center y_center bbox_width bbox_height
                
                class_id, x_center, y_center, bbox_width, bbox_height = parts
                bboxes.append([class_id, x_center, y_center, bbox_width, bbox_height])
            else:
                print("Invalid bounding box format: ", line)
        except ValueError:
            print(f"Error reading line: {line.strip()}")
           
    return np.array(bboxes)



def load_file_names(image_dir, label_dir):
    """
    Load all images and their corresponding label files based on the list of image names.

    Input:
        - folder_path (str): Path to the dataset directory, either to rgb or lidar
        - image_names (list): List of image names
    Output:
        - images (list): List of loaded images in RGB format
        - label_files (list): List of corresponding label files
    """

    # if "lidar" in folder_path:
    #     image_type = "combined_color"
    # elif "rgb" in folder_path:
    #     image_type = "images"
    # else:
    #     raise ValueError("Invalid path name. Please provide a valid image name.")
   
    # image_path = os.path.join(folder_path, image_type, dataset_type)
    # label_path = os.path.join(folder_path, "labels", dataset_type)    

    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))
    
    image_names = [filename for filename in image_files]
    label_files = [filename for filename in label_files]
    return image_names, label_files

def load_images_and_labels(image_dir, label_dir):
	# if "lidar" in folder_path:
	#     image_type = "combined_color"
	# elif "rgb" in folder_path:
	#     image_type = "images"
	# else:
	#     raise ValueError("Invalid path name. Please provide a valid image name.")
	image_names, label_names = load_file_names(image_dir, label_dir)
	data_dict = {}

	for image_name, label_name in zip(image_names, label_names):
		image_path = os.path.join(image_dir, image_name)
		label_path = os.path.join(label_dir, label_name)

		image = cv2.imread(image_path)
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		bbox = get_bounding_boxes(label_path)[:,1:]
		label = get_bounding_boxes(label_path)[:,0]
		dict_index = os.path.splitext(label_name)[0]
		data_dict[dict_index] = {
			"image": image_rgb,
			"bbox": bbox,
			"label": label
		}

	return data_dict

# Create a dataset class to load the images and labels from the folder 
# class Dataset(torch.utils.data.Dataset): 
# 	"""
#     This class with following functions is gathered from Geeks for Geeks: https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/ 
#     Accessed: 09-04-2025
#     """
# 	def __init__( 
# 		self, image_dir, label_dir, 
# 		image_size=640, num_classes=1, transform=None
# 	): 
# 		# Read the csv file with image names and labels 
# 		self.data_dict = load_images_and_labels(image_dir, label_dir)
# 		self.label_list = list(self.data_dict.keys())
# 		# Image and label directories 
# 		self.image_dir = image_dir 
# 		self.label_dir = label_dir 
# 		# Image size 
# 		self.image_size = image_size 
# 		# Transformations 
# 		self.transform = transform
# 		# Number of classes 
# 		self.num_classes = num_classes 
# 		# Ignore IoU threshold 
# 		self.ignore_iou_thresh = 0.5
# 		print(f"Available keys in data_dict: {list(self.data_dict.keys())}")
# 	def __len__(self): 
# 		return len(self.label_list) 
	
# 	def __getitem__(self, idx):
		
# 		dict_index = os.path.splitext(self.label_list[idx])[0]
# 		data = self.data_dict[dict_index]
            
# 		transformed = self.transform(
# 			image=data[dict_index]["image"],
# 			bboxes=data[dict_index]["bbox"],
# 			labels=data[dict_index]["label"]
# 		)

#     	# Extract the transformed components
# 		image = transformed["image"]
# 		bboxes = transformed["bboxes"]
# 		labels = transformed["labels"]

# 		print(f"Image shape: {image.shape}")

# 		return image, {
# 			"bboxes": torch.tensor(bboxes, dtype=torch.float32),
# 			"label": torch.tensor(labels, dtype=torch.float32)
# 		}
        
class Dataset(torch.utils.data.Dataset):
	"""
    This class with following functions is gathered from Geeks for Geeks: https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/ 
    Accessed: 09-04-2025
    """

	def __init__(
			self, image_dir, label_dir,
			image_size=640,
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
		# for box in yolo_boxes:
		# 	x_center, y_center, width, height, class_label = box

		# 	# Resize box coordinates to match transform size (300x300)
		# 	x_min = (x_center - width / 2)*img_width
		# 	y_min = (y_center - height / 2)*img_height
		# 	x_max = (x_center + width / 2)*img_width
		# 	y_max = (y_center + height / 2)*img_height

		# 	#scale_x = 300 / img_width
		# 	#scale_y = 300 / img_height

		# 	#x_min = (x_center - width / 2) * img_width * scale_x
		# 	#y_min = (y_center - height / 2) * img_height * scale_y
		# 	#x_max = (x_center + width / 2) * img_width * scale_x
		# 	#y_max = (y_center + height / 2) * img_height * scale_y

		# 	#x_min = x_min / 300
		# 	#y_min = y_min / 300
		# 	#x_max = x_max / 300
		# 	#y_max = y_max / 300

		# 	boxes.append([x_min, y_min, x_max, y_max])
		# 	#labels.append(int(class_label))  # usually 0 or 1 for binary
		# 	mapped_label = 1 if int(class_label) == 0 else int(class_label)
		# 	labels.append(int(mapped_label))


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

		# Clamp boxes to minimum size to avoid zero/near-zero width or height
		MIN_SIZE = 1.0  # pixel minimum
		print(f"Image shape: {image.shape}, Target: {target}")
		# boxes_clamped = []
		# for b in boxes:
		# 	x_min, y_min, x_max, y_max = b
		# 	width = max(x_max - x_min, MIN_SIZE)
		# 	height = max(y_max - y_min, MIN_SIZE)

		# 	# Recompute x_max/y_max to match clamped size
		# 	x_max = x_min + width
		# 	y_max = y_min + height

		# 	boxes_clamped.append([x_min, y_min, x_max, y_max])
			
		# boxes = torch.tensor(boxes_clamped, dtype=torch.float32)
		boxes = torch.tensor(boxes, dtype=torch.float32)
		labels = torch.tensor(labels, dtype=torch.int64)

		# Convert normalized VOC boxes to absolute pixel coords (300x300)
		boxes = np.array(boxes)  # shape: (N, 4)
		# boxes[:, [0, 2]] *= self.image_size  # x_min, x_max
		# boxes[:, [1, 3]] *= self.image_size  # y_min, y_max

		# Ensure non-empty, well-formed tensors
		if len(boxes) == 0:
			boxes = torch.zeros((0, 4), dtype=torch.float32)
			labels = torch.zeros((0,), dtype=torch.int64)
		else:
			boxes = torch.tensor(boxes, dtype=torch.float32)
			labels = torch.tensor(labels, dtype=torch.int64)

		target = {
			'boxes': boxes,
			'labels': labels,
			'orig_size': torch.tensor([img_height, img_width], dtype=torch.int32)
		}

		return image, target