import numpy as np
import matplotlib.pyplot as plt


def calculate_iou(prediction_box, gt_box):
    """
    From Assignment 4
    Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax] - > must convert from [x_center,y_center,w,h]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    # Compute intersection

    # Compute union
    intersection = max(0, min(prediction_box[2], gt_box[2]) - max(prediction_box[0], gt_box[0]))*max(0, min(prediction_box[3], gt_box[3]) - max(prediction_box[1], gt_box[1]))
    union = (prediction_box[2] - prediction_box[0])*(prediction_box[3] - prediction_box[1]) + (gt_box[2] - gt_box[0])*(gt_box[3] - gt_box[1]) - intersection
    

    epsilon = 1e-6
    iou = intersection/ (union + epsilon)  # Avoid division by zero
    #END OF YOUR CODE

    assert iou >= 0 and iou <= 1
    return iou