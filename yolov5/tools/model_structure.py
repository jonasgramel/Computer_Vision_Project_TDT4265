import torch
import yaml
from torchvision.ops import box_iou
import torch.nn as nn
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, classes=1)

detec_layer = model.model[-1]
anchors = detec_layer.anchors.clone().detach()
strides = detec_layer.stride
scaled_anchors = anchors * strides.view(-1, 1, 1)

image_size = 640
# Freeze backbone
for i, m in enumerate(model.model):
    if i <= 9:
        for p in m.parameters():
            p.requires_grad = False

# Test if layers are frozed correctly
for i, (name, m) in enumerate(model.named_modules()):
    if hasattr(m, "weight") and hasattr(m.weight, "requires_grad"):
        print(f"{i:02d} | {name:40s} | requires_grad: {m.weight.requires_grad}")


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
		ious = box_iou(box_preds[obj], target[..., 1:5][obj]).detach() 
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
