import torch
import torch.nn as nn

from utils.metrics_calculation import iou

# Non-maximum suppression function to remove overlapping bounding boxes 
def nms(bboxes, iou_threshold, threshold): 
    """
    This function is gathered from Geeks for Geeks: https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/ 
    Accessed: 09-04-2025
    """
    # Filter out bounding boxes with confidence below the threshold. 
    bboxes = [box for box in bboxes if box[1] > threshold] 
  
    # Sort the bounding boxes by confidence in descending order. 
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) 
  
    # Initialize the list of bounding boxes after non-maximum suppression. 
    bboxes_nms = [] 
  
    while bboxes: 
        # Get the first bounding box. 
        first_box = bboxes.pop(0) 
  
        # Iterate over the remaining bounding boxes. 
        for box in bboxes: 
        # If the bounding boxes do not overlap or if the first bounding box has 
        # a higher confidence, then add the second bounding box to the list of 
        # bounding boxes after non-maximum suppression. 
            if box[0] != first_box[0] or iou( 
                torch.tensor(first_box[2:]), 
                torch.tensor(box[2:]), 
            ) < iou_threshold: 
                # Check if box is not in bboxes_nms 
                if box not in bboxes_nms: 
                    # Add box to bboxes_nms 
                    bboxes_nms.append(box) 
  
    # Return bounding boxes after non-maximum suppression. 
    return bboxes_nms


# Defining YOLO loss class 
class YOLOLoss(nn.Module): 
	def __init__(self): 
		super().__init__() 
		self.mse = nn.MSELoss() 
		self.bce = nn.BCEWithLogitsLoss() 
		self.cross_entropy = nn.CrossEntropyLoss() 
		self.sigmoid = nn.Sigmoid() 
	
	def forward(self, pred, target, anchors): 
		# Identifying which cells in target have objects 
		# and which have no objects 
		obj = target[..., 0] == 1
		no_obj = target[..., 0] == 0

		# Calculating No object loss 
		no_object_loss = self.bce( 
			(pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]), 
		) 

		
		# Reshaping anchors to match predictions 
		anchors = anchors.reshape(1, 3, 1, 1, 2) 
		# Box prediction confidence 
		box_preds = torch.cat([self.sigmoid(pred[..., 1:3]), 
							torch.exp(pred[..., 3:5]) * anchors 
							],dim=-1) 
		# Calculating intersection over union for prediction and target 
		ious = iou(box_preds[obj], target[..., 1:5][obj]).detach() 
		# Calculating Object loss 
		object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]), 
							ious * target[..., 0:1][obj]) 

		
		# Predicted box coordinates 
		pred[..., 1:3] = self.sigmoid(pred[..., 1:3]) 
		# Target box coordinates 
		target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors) 
		# Calculating box coordinate loss 
		box_loss = self.mse(pred[..., 1:5][obj], 
							target[..., 1:5][obj]) 

		
		# Claculating class loss 
		class_loss = self.cross_entropy((pred[..., 5:][obj]), 
								target[..., 5][obj].long()) 

		# Total loss 
		return ( 
			box_loss 
			+ object_loss 
			+ no_object_loss 
			+ class_loss 
		)
