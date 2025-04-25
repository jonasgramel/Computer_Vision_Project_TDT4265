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


