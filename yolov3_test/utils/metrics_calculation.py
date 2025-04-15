import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.box_preparation import yolo_to_xy_coords

# Defining a function to calculate Intersection over Union (IoU) 
def iou(box1, box2, is_pred=True): 
    """
    This function is gathered from Geeks for Geeks: https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/ 
    Accessed: 09-04-2025
    """
    if is_pred: 
        # IoU score for prediction and label 
        # box1 (prediction) and box2 (label) are both in [x, y, width, height] format 
          
        # Box coordinates of prediction 
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2
  
        # Box coordinates of ground truth 
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2
  
        # Get the coordinates of the intersection rectangle 
        x1 = torch.max(b1_x1, b2_x1) 
        y1 = torch.max(b1_y1, b2_y1) 
        x2 = torch.min(b1_x2, b2_x2) 
        y2 = torch.min(b1_y2, b2_y2) 
        # Make sure the intersection is at least 0 
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) 
  
        # Calculate the union area 
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1)) 
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1)) 
        union = box1_area + box2_area - intersection 
  
        # Calculate the IoU score 
        epsilon = 1e-6
        iou_score = intersection / (union + epsilon) 
  
        # Return IoU score 
        return iou_score 
      
    else: 
        # IoU score based on width and height of bounding boxes 
          
        # Calculate intersection area 
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * torch.min(box1[..., 1], box2[..., 1]) 
  
        # Calculate union area 
        box1_area = box1[..., 0] * box1[..., 1] 
        box2_area = box2[..., 0] * box2[..., 1] 
        union_area = box1_area + box2_area - intersection_area 
  
        # Calculate IoU score 
        iou_score = intersection_area / union_area 
  
        # Return IoU score 
        return iou_score
    

def calculate_precision(num_tp, num_fp, num_fn):
    """ 
    From Assignment 4
    Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    # YOUR CODE HERE
    precision = num_tp/(num_tp + num_fp) if (num_tp + num_fp) != 0 else 1
    return precision
    #END OF YOUR CODE
    raise NotImplementedError


def calculate_recall(num_tp, num_fp, num_fn):
    
    """ 
    From Assignment 4
    Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    # YOUR CODE HERE
    recall = num_tp/(num_tp + num_fn) if (num_tp + num_fn) != 0 else 0
    return recall
    #END OF YOUR CODE
    raise NotImplementedError


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """
    From Assignment 4
    Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # YOUR CODE HERE

    # Find all possible matches with a IoU >= iou threshold
    iou_list = []
    for i, prediction_box in enumerate(prediction_boxes):
        for j, gt_box in enumerate(gt_boxes):
            # print("Pred shape", prediction_box.shape)
            # print("GT shape", gt_box.shape)
            # print("Pred box[:4]", prediction_box[:4])
            # print("GT box[:4]", gt_box[:4])
            iou = iou(prediction_box[:4], gt_box[:4])
            if iou >= iou_threshold:
                iou_list.append((iou, i, j))
            # print("IoU: ", iou)
    # Sort all matches on IoU in descending order
    iou_list.sort(reverse=True, key=lambda x: x[0])

    # Find all matches with the highest IoU threshold
    filtered_prediction_boxes = []
    filtered_gt_boxes = []
    used_prediction_indices = []
    used_gt_indices = []

    for iou, i, j in iou_list:
        if i not in used_prediction_indices and j not in used_gt_indices:
            filtered_prediction_boxes.append(prediction_boxes[i][:4])
            filtered_gt_boxes.append(gt_boxes[j][:4])
            used_prediction_indices.append(i)
            used_gt_indices.append(j)
    # print(f"Matched {len(filtered_prediction_boxes)} boxes (IoU >= {iou_threshold})")

    #print("Sample IoUs:")
    #for iou, _, _ in iou_list[:5]:
    #    print(f"{iou:.2f}")

    return np.array(filtered_prediction_boxes), np.array(filtered_gt_boxes)
    #END OF YOUR CODE



def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """
    From Assignment 4
    Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    
    matched_prediction_boxes, matched_gt_boxes = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    num_tp = len(matched_prediction_boxes)
    num_fp = len(prediction_boxes) - num_tp
    num_fn = len(gt_boxes) - num_tp

    return {"true_pos": num_tp, "false_pos": num_fp, "false_neg": num_fn}
    #END OF YOUR CODE

    raise NotImplementedError


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """
    From Assignment 4
    Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    # YOUR CODE HERE
    num_tp = 0
    num_fp = 0
    num_fn = 0

    for prediction_boxes, gt_boxes in zip(all_prediction_boxes, all_gt_boxes):
        results = calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold)
        num_tp += results["true_pos"]
        num_fp += results["false_pos"]
        num_fn += results["false_neg"]
    
    precision = calculate_precision(num_tp, num_fp, num_fn)
    recall = calculate_recall(num_tp, num_fp, num_fn)

    return (precision, recall)
    #END OF YOUR CODE

    raise NotImplementedError


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """
    From Assignment 4
    Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)

    precisions = []
    recalls = []

    for threshold in confidence_thresholds:
        filtered_predictions = []

        for prediction_boxes, scores in zip(all_prediction_boxes, confidence_scores):
            mask = scores >= threshold
            filtered_predictions.append(prediction_boxes[mask] if len(prediction_boxes) >= 0 else np.array([]))

        results = calculate_precision_recall_all_images(filtered_predictions, all_gt_boxes, iou_threshold)

        precisions.append(results[0])
        recalls.append(results[1])
    # END OF YOUR CODE
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("fiugres/precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    average_precision = 0
    for r in recall_levels:
        valid_precisions = precisions[recalls >= r]
        if valid_precisions.size > 0: # checks if valid_precisions is an empty array
            average_precision += np.max(precisions[recalls >= r])
        else:
            continue
    average_precision /= len(recall_levels) # average over the recall levels
    #END OF YOUR CODE

    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (array): yolo format
        predicted_boxes: (array): yolo format
    Returns:
        precisions, recalls, mean_average_precision: (np.array of floats) length of N
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for i in range(len(ground_truth_boxes)):
        predicted_boxes = yolo_to_xy_coords(predicted_boxes[i])
        gt_boxes = yolo_to_xy_coords(ground_truth_boxes[i])
        confidence_scores.append(predicted_boxes[:, 1])

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))
    return precisions, recalls, mean_average_precision
