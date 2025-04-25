import torch
import yaml
from torchvision.ops import box_iou
import torch.nn as nn
import torch.nn.functional as F

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
# for i, (name, m) in enumerate(model.named_modules()):
# 	if hasattr(m, "weight") and hasattr(m.weight, "requires_grad"):
# 		print(f"{i:02d} | {name:40s} | requires_grad: {m.weight.requires_grad}")

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
        device = predictions[0].device
        loss_cls = 0
        loss_obj = 0
        loss_box = 0
        BCEcls = nn.BCEWithLogitsLoss()
        BCEobj = nn.BCEWithLogitsLoss()
        MSEbox = nn.MSELoss()

        for scale_idx, pred in enumerate(predictions):
            B, _, H, W = pred.shape
            A = scaled_anchors[scale_idx].shape[0]
            pred = pred.view(B, A, self.num_classes + 5, H, W).permute(0, 1, 3, 4, 2)

            # Create targets for this scale
            obj_mask = torch.zeros((B, A, H, W), dtype=torch.bool, device=device)
            noobj_mask = torch.ones((B, A, H, W), dtype=torch.bool, device=device)
            tx = torch.zeros((B, A, H, W), device=device)
            ty = torch.zeros((B, A, H, W), device=device)
            tw = torch.zeros((B, A, H, W), device=device)
            th = torch.zeros((B, A, H, W), device=device)
            tcls = torch.zeros((B, A, H, W, self.num_classes), device=device)

            anchor_grid = scaled_anchors[scale_idx].to(device)  # [A, 2]

            for b in range(B):
                boxes = targets[b]['boxes'] * torch.tensor([W, H, W, H], device=device)
                labels = targets[b]['labels'].long()
                for box_idx, box in enumerate(boxes):
                    gx, gy, gw, gh = box
                    gi = int(gx)
                    gj = int(gy)

                    if gi >= W or gj >= H:
                        continue  # Skip boxes outside grid

                    # Get best matching anchor
                    wh = box[2:].unsqueeze(0)
                    anchor_ratios = anchor_grid / wh
                    ious = torch.min(anchor_ratios, 1. / anchor_ratios).prod(1)
                    best_anchor = ious.argmax()

                    obj_mask[b, best_anchor, gj, gi] = 1
                    noobj_mask[b, best_anchor, gj, gi] = 0

                    tx[b, best_anchor, gj, gi] = gx - gi
                    ty[b, best_anchor, gj, gi] = gy - gj
                    tw[b, best_anchor, gj, gi] = torch.log(gw / anchor_grid[best_anchor][0] + 1e-16)
                    th[b, best_anchor, gj, gi] = torch.log(gh / anchor_grid[best_anchor][1] + 1e-16)

                    tcls[b, best_anchor, gj, gi, labels[box_idx]] = 1

            pred_boxes = pred[..., 0:4]
            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]

            loss_box += MSEbox(pred_boxes[..., 0][obj_mask], tx[obj_mask])
            loss_box += MSEbox(pred_boxes[..., 1][obj_mask], ty[obj_mask])
            loss_box += MSEbox(pred_boxes[..., 2][obj_mask], tw[obj_mask])
            loss_box += MSEbox(pred_boxes[..., 3][obj_mask], th[obj_mask])

            loss_obj += BCEobj(pred_obj[obj_mask], torch.ones_like(pred_obj[obj_mask]))
            loss_obj += BCEobj(pred_obj[noobj_mask], torch.zeros_like(pred_obj[noobj_mask]))

            if self.num_classes > 1:
                loss_cls += BCEcls(pred_cls[obj_mask], tcls[obj_mask])

        total_loss = (
            self.lambda_box * loss_box +
            self.lambda_obj * loss_obj +
            self.lambda_cls * loss_cls
        )

        return total_loss
