import torch
from torchvision.models.detection import ssd300_vgg16

model = ssd300_vgg16(pretrained=True, num_classes=1)
model.eval()
