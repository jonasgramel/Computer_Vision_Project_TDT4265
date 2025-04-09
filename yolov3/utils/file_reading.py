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

            # Calculate the top-left and bottom-right corners of the bounding box
            x_min, y_min = int(x_center - bbox_width / 2), int(y_center - bbox_height / 2)
            x_max, y_max = int(x_center + bbox_width / 2), int(y_center + bbox_height / 2)
            bboxes_xy[i] = [x_min, y_min, x_max, y_max]
    return bboxes_xy

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
