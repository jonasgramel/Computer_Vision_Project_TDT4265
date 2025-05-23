import torch
import numpy as np
# Function to convert cells to bounding boxes 
def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True): 
    """
    This function is gathered from Geeks for Geeks: https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/ 
    Accessed: 09-04-2025
    """
    # Batch size used on predictions 
    batch_size = predictions.shape[0] 
    # Number of anchors 
    num_anchors = len(anchors) 
    # List of all the predictions 
    box_predictions = predictions[..., 1:5] 
  
    # If the input is predictions then we will pass the x and y coordinate 
    # through sigmoid function and width and height to exponent function and 
    # calculate the score and best class. 
    if is_predictions: 
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2) 
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2]) 
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors 
        scores = torch.sigmoid(predictions[..., 0:1]) 
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1) 
      
    # Else we will just calculate scores and best class. 
    else: 
        scores = predictions[..., 0:1] 
        best_class = predictions[..., 5:6] 
  
    # Calculate cell indices 
    cell_indices = ( 
        torch.arange(s) 
        .repeat(predictions.shape[0], 3, s, 1) 
        .unsqueeze(-1)
        .to(predictions.device) 
    ) 
  
    # Calculate x, y, width and height with proper scaling 
    x = 1 / s * (box_predictions[..., 0:1] + cell_indices) 
    y = 1 / s * (box_predictions[..., 1:2] +
                 cell_indices.permute(0, 1, 3, 2, 4)) 
    width_height = 1 / s * box_predictions[..., 2:4] 
  
    # Concatinating the values and reshaping them in 
    # (BATCH_SIZE, num_anchors * S * S, 6) shape 
    # converted_bboxes = torch.cat( 
    #     (best_class, scores, x, y, width_height), dim=-1
    # ).reshape(batch_size, num_anchors * s * s, 6)
    converted_bboxes = torch.cat( 
        (best_class, scores, x, y, width_height), dim=-1
    ).reshape(batch_size, num_anchors * s * s, 6) 
  
    # Returning the reshaped and converted bounding box list 
    return converted_bboxes.tolist()

def yolo_to_xy_coords(bboxes):
    """
    Input: 
        - bboxes (numpy array): Array of bounding boxes in YOLO format: [class_id, x_center, y_center, bbox_width, bbox_height]

    Output:
        - bboxes (numpy array): Array of bounding boxes in pixel format: [x_min, y_min, x_max, y_max]
    """
    bboxes_xy = np.zeros((len(bboxes), 4), dtype=int)
    height, width = 416, 416
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