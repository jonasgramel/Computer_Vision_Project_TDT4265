import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
import torchvision.transforms.functional as TF

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
                bboxes.append([x_center, y_center, bbox_width, bbox_height])
            else:
                print("Invalid bounding box format: ", line)
        except ValueError:
            print(f"Error reading line: {line.strip()}")
           
    return np.array(bboxes)


def yolo_to_xy_coords(image, bboxes):
    """
    Input: 
        - bboxes (numpy array): Array of bounding boxes in YOLO format: [class_id, x_center, y_center, bbox_width, bbox_height]

    Output:
        - bboxes (numpy array): Array of bounding boxes in pixel format: [x_min, y_min, x_max, y_max]
    """
    bboxes_xy = np.zeros((len(bboxes), 4), dtype=int)
    height, width = image.shape[:2]
    for i, bbox in enumerate(bboxes):
        if bbox is not None:       # Check if boundings boxes present
            x_center, y_center, bbox_width, bbox_height = bbox

            # Convert YOLO format to pixel coordinates
            x_center, y_center = int(x_center * width), int(y_center * height)
            bbox_width, bbox_height = int(bbox_width * width), int(bbox_height * height)

            # Calculate the bottom-left and top-right corners of the bounding box
            x_min, y_min = int(x_center - bbox_width / 2), int(y_center - bbox_height / 2)
            x_max, y_max = int(x_center + bbox_width / 2), int(y_center + bbox_height / 2)
            bboxes_xy[i] = [x_min, y_min, x_max, y_max]
    return bboxes_xy

def xy_to_yolo_coords(bboxes):
    """
    Convert bounding boxes from pixel coordinates to YOLO format.

    Input:
        - bboxes (numpy array): Array of bounding boxes in pixel format: [x_min, y_min, x_max, y_max]

    Output:
        - bboxes (numpy array): Array of bounding boxes in YOLO format: [class_id, x_center, y_center, bbox_width, bbox_height]
    """
    bboxes_yolo = np.zeros((len(bboxes), 5), dtype=float)
    for i, bbox in enumerate(bboxes):
        if bbox is not None:       # Check if boundings boxes present
            x_min, y_min, x_max, y_max = bbox

            # Calculate the center and size of the bounding box
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            bbox_width = (x_max - x_min) 
            bbox_height = (y_max - y_min)

            bboxes_yolo[i] = [0.0, x_center, y_center, bbox_width, bbox_height]  # class_id is set to 0 for all boxes
    return bboxes_yolo

def draw_bounding_boxes(image, bboxes):
    """
    Based on Listing 1 in Bavirisetti et al. (2025): https://www.sciencedirect.com/science/article/pii/S2352340925001350?via%3Dihub
    Draw bounding boxes on the image.

    Input: 
        - image (numpy array): Image to draw bounding boxes on
        - bboxes (numpy array): Array of bounding boxes in YOLO format
    Output:
        - image (numpy array): Image with bounding boxes drawn
    """
    height, width = image.shape[:2]
    for bbox in bboxes:
        if bbox is not None:       # Check if boundings boxes present
            x_center, y_center, bbox_width, bbox_height = bbox

            # Convert YOLO format to pixel coordinates
            x_center, y_center = int(x_center * width), int(y_center * height)
            bbox_width, bbox_height = int(bbox_width * width), int(bbox_height * height)

            # Calculate the top-left and bottom-right corners of the bounding box
            x1, y1 = int(x_center - bbox_width / 2), int(y_center - bbox_height / 2)
            x2, y2 = int(x_center + bbox_width / 2), int(y_center + bbox_height / 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, "Pole", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)
    return image

def load_file_names(folder_path, dataset_type):
    """
    Load all images and their corresponding label files based on the list of image names.

    Input:
        - folder_path (str): Path to the dataset directory, either to rgb or lidar
        - image_names (list): List of image names
    Output:
        - images (list): List of loaded images in RGB format
        - label_files (list): List of corresponding label files
    """

    if "lidar" in folder_path:
        image_type = "combined_color"
    elif "rgb" in folder_path:
        image_type = "images"
    else:
        raise ValueError("Invalid path name. Please provide a valid image name.")
   
    image_path = os.path.join(folder_path, image_type, dataset_type)
    label_path = os.path.join(folder_path, "labels", dataset_type)    

    image_files = sorted(os.listdir(image_path))
    label_files = sorted(os.listdir(label_path))
    
    image_names = [filename for filename in image_files]
    label_files = [filename for filename in label_files]
    return image_names, label_files

def load_images_and_labels(folder_path, input_size, dataset_type):
    if "lidar" in folder_path:
        image_type = "combined_color"
    elif "rgb" in folder_path:
        image_type = "images"
    else:
        raise ValueError("Invalid path name. Please provide a valid image name.")
    image_names, label_names = load_file_names(folder_path, dataset_type)
    image_arr = []
    label_arr = []

    for image_name, label_name in zip(image_names, label_names):
        image_path = os.path.join(folder_path, image_type, dataset_type, image_name)
        label_path = os.path.join(folder_path, "labels", dataset_type, label_name)

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox = get_bounding_boxes(label_path)
        
        image_resized, bbox_resized = resize_image(image_rgb, bbox, input_size)
        image_arr.append(image_resized)
        label_arr.append(bbox_resized)

    return image_arr, label_arr, image_names
