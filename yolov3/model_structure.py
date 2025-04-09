import torch
import torchvision.models as models

image_size = (416, 416)  # Image size for YOLOv3

backbone = models.darknet53(weights="DEFAULT")

class YOLOv3(torch.nn.Module):
    def __init__(self, backbone, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.backbone = backbone
        # Define the layers for YOLOv3
       
    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x    