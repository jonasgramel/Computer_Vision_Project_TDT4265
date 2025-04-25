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


import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes=1, strides=[8, 16, 32], lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5):
        super().__init__()
        self.anchors = anchors  # shape: (3, 3, 2)
        self.num_classes = num_classes
        self.strides = strides
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

    def forward(self, predictions, targets, scaled_anchors):
        # predictions = list of 3 outputs (1 for each scale), each [B, A*(5+C), H, W]
        loss_box = 0
        loss_obj = 0
        loss_cls = 0

        for i in range(3):  # for each scale
            pred = predictions[i]
            B, _, H, W = pred.shape
            A = len(scaled_anchors[i])
            pred = pred.view(B, A, self.num_classes + 5, H, W).permute(0, 1, 3, 4, 2).contiguous()

            # Placeholder: create target tensors (this is the hard part — match GT boxes to anchors here!)
            # Example: t_box, t_obj, t_cls = build_targets(pred, targets, scaled_anchors[i])
            # For now, we assume you have target tensors

            # Calculate each part (dummy below)
            # loss_box += F.mse_loss(pred[..., 0:4], t_box)
            # loss_obj += F.binary_cross_entropy_with_logits(pred[..., 4], t_obj)
            # loss_cls += F.binary_cross_entropy_with_logits(pred[..., 5:], t_cls)

            pass  # ← implement or plug in your target matching logic here

        total_loss = self.lambda_box * loss_box + self.lambda_obj * loss_obj + self.lambda_cls * loss_cls
        return total_loss
