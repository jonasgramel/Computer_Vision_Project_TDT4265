
import torch
from mAP import calculate_iou
from file_reading import xy_to_yolo_coords

def multibox_prior(data, sizes, ratios):
    """
    From Zhang et al. Dive into Deep Learning, 2023, Cambridge University Press

    Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height # Scaled steps in y axis
    steps_w = 1.0 / in_width # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
        sizes[0] * torch.sqrt(ratio_tensor[1:])))\
        * in_height / in_width # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
    sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
        dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """
    From Zhang et al. Dive into Deep Learning, 2023, Cambridge University Press
    Assign closest ground-truth bounding boxes to anchor boxes.
    Input:
        - ground_truth: tensor of shape (N, 4) containing the coordinates of
            the ground-truth bounding boxes. [xmin, ymin, xmax, ymax]
        - anchors: tensor of shape (M, 4) containing the coordinates of the
            anchor boxes. [xmin, ymin, xmax, ymax]
        - device: device to which the tensors should be moved.
        - iou_threshold: IoU threshold for assigning a ground-truth bounding box
            to an anchor box.
    """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = calculate_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
    device=device)
    
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard) # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """
    From Zhang et al. Dive into Deep Learning, 2023, Cambridge University Press
    Transform for anchor box offsets.
    Input:
        - anchors: tensor of shape (N, 4) containing the coordinates of the
            anchor boxes. [xmin, ymin, xmax, ymax]
        - assigned_bb: tensor of shape (M, 4) containing the coordinates of the
            assigned ground-truth bounding boxes. [xmin, ymin, xmax, ymax]
    Output:
        - offset: tensor of shape (N, 4) containing the offsets for the anchor
            boxes. [dx, dy, dw, dh]
    """
    c_anc = xy_to_yolo_coords(anchors)[::, 1:]
    c_assigned_bb = xy_to_yolo_coords(assigned_bb)[::,1:]
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset