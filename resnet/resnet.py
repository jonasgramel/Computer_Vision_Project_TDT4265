import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.models as models
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from utils.box_preparation import convert_cells_to_bboxes
from utils.file_reading import Dataset
from utils.metrics_calculation import iou_loss, compute_iou, FocalLoss
from utils.plot_picture import plot_image, visualize_dataset_sample, visualize_predictions, visualize_preds_vs_gt

import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 
import matplotlib.pyplot as plt
import numpy as np
np.float = float


num_classes = 2

device = torch.device('cuda')
# Learning rate for training 
learning_rate = 1e-5

# Number of epochs for training 
num_epochs = 20
unfreeze_at_epoch = 10

# Image size 
image_size = 224

batch_size=8

# Define custom anchor generator with narrow/tall aspect ratios
anchor_generator = AnchorGenerator(
    sizes=((32,), (64,), (128,), (256,), (512,)),  # 5 tuples for 5 FPN levels
    aspect_ratios=((0.5, 1.0, 2.0),) * 5  # repeat aspect ratios 5 times
)

# Define ResNet50 backbone with FPN
backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(backbone_name='resnet50', weights='DEFAULT')

# Initialize the Faster R-CNN model
pretrained_model = FasterRCNN(
    backbone=backbone,
    num_classes=num_classes,  # 1 class (snow-pole) + 1 background
    rpn_anchor_generator=anchor_generator
)

# Freeze layers except layer4 of ResNet50 initially
for name, param in pretrained_model.backbone.body.named_parameters():
    if "layer4" not in name:  # Freeze everything except layer4
        param.requires_grad = False

in_features = pretrained_model.roi_heads.box_predictor.cls_score.in_features
pretrained_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)


optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, pretrained_model.parameters()),
    lr=1e-5, weight_decay=0.0005
)

# Transform for training 
train_transform = A.Compose( 
	[ 
		# Rescale an image so that maximum side is equal to image_size 
		A.LongestMaxSize(max_size=image_size), 
		# Pad remaining areas with zeros 
		A.PadIfNeeded( 
			min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
		), 
		# Random color jittering 
		A.ColorJitter( 
			brightness=0.5, contrast=0.5, 
			saturation=0.5, hue=0.5, p=0.5
		), 
		# Flip the image horizontally 
		A.HorizontalFlip(p=0.5), 
		# Normalize the image 
		A.Normalize( 
			mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255
		), 
		# Convert the image to PyTorch tensor 
		ToTensorV2() 
	], 
	# Augmentation for bounding boxes 
	bbox_params=A.BboxParams( 
					format="pascal_voc", 
					min_visibility=0.2, 
					label_fields=[] 
				) 
) 


val_transform = A.Compose( 
	[ 
		# Rescale an image so that maximum side is equal to image_size 
		A.LongestMaxSize(max_size=image_size), 
		# Pad remaining areas with zeros 
		A.PadIfNeeded( 
			min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
		), 
		# Normalize the image 
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255,
        ),
		# Convert the image to PyTorch tensor 
		ToTensorV2() 
	], 
	# Augmentation for bounding boxes 
	bbox_params=A.BboxParams(
					format="pascal_voc", 
					min_visibility=0.4, 
					label_fields=[] 
				)
)

def collate_fn(batch):
    return tuple(zip(*batch))

# Creating a dataset object 
dataset = Dataset( 
	#csv_file="train.csv", 
	image_dir="/work/datasets/tdt4265/ad/open/Poles/lidar/combined_color/train", 
	label_dir="/work/datasets/tdt4265/ad/open/Poles/lidar/labels/train", 
	transform=train_transform 
) 

# Creating a dataloader object 
loader = torch.utils.data.DataLoader( 
	dataset=dataset, 
	batch_size=batch_size, 
	shuffle=True, 
    collate_fn=collate_fn 

)

val_dataset = Dataset(
    image_dir="/work/datasets/tdt4265/ad/open/Poles/lidar/combined_color/valid", 
    label_dir="/work/datasets/tdt4265/ad/open/Poles/lidar/labels/valid", 
    transform=val_transform
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=collate_fn
)

def filter_predictions(pred, score_thresh=0.2):
    boxes = pred['boxes']
    scores = pred['scores']
    labels = pred['labels']

    keep = scores > score_thresh
    return {
        'boxes': boxes[keep],
        'scores': scores[keep],
        'labels': labels[keep]
    }

images, targets = next(iter(loader))
print(targets[0])

trainresnet = True

if __name__ == "__main__":
    
    pretrained_model.to(device)

    if trainresnet:
        
        print("♫Training Montage♫ by Vince DiCola starts playing...")
        train_losses = []
        train_losses = []
        val_losses = []  # To track validation losses
        val_mAP = []  # To track validation mAP
        val_mAP050 = []  # To track validation mAP


        for i in range(3):
            visualize_dataset_sample(dataset, idx=i)

        for epoch in tqdm.trange(num_epochs):

            if epoch < 2:
                lr = 1e-6
                for g in optimizer.param_groups:
                    g['lr'] = lr
            if epoch == 2:  # Unfreeze 'layer3'
                print("Unfreezing layer 3. Cool party!")
                torch.cuda.empty_cache()
                lr = 1e-5
                for g in optimizer.param_groups:
                    g['lr'] = lr
                for param in pretrained_model.backbone.body.layer3.parameters():
                    param.requires_grad = True
            elif epoch == 6:  # Unfreeze 'layer2'
                print("Unfreezing layer 2. What killed the dinosaurs? The Ice Age!")
                torch.cuda.empty_cache()
                for param in pretrained_model.backbone.body.layer2.parameters():
                    param.requires_grad = True
            elif epoch == 10:  # Unfreeze the entire backbone
                torch.cuda.empty_cache()
                print("Unfreezing the entire backbone. Everybody chill!")
                for param in pretrained_model.backbone.parameters():
                    param.requires_grad = True
            
            pretrained_model.train()

            epoch_loss = 0

            for images, labels in loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in labels] 

                for t in targets:

                    if torch.any(t['labels'] >= num_classes):
                        print("Label out of range:", t['labels'])
                        raise ValueError("Target label exceeds num_classes - 1")
                    
                    if not torch.isfinite(t['boxes']).all():
                        print("invalid box detected:", t['boxes'])

                    if not torch.isfinite(t['labels']).all():
                        print("invalid label detected:", t['labels'])             


                loss_dict = pretrained_model(images, targets)
                for k, v in loss_dict.items():
                    if not torch.isfinite(v):
                        print(f"🚨 Non-finite {k} loss: {v}")
                        raise RuntimeError("Loss exploded")

                    if v.item() > 1000:
                        print(f"🔥 Suspiciously large {k} loss: {v.item()}")

                # Extract the losses from loss_dict
                loss_classifier = loss_dict['loss_classifier']  # Original classification loss
                loss_box_reg = loss_dict['loss_box_reg']  # Bounding box regression loss
                loss_objectness = loss_dict['loss_objectness']  # RPN objectness loss
                loss_rpn_box_reg = loss_dict['loss_rpn_box_reg']  # RPN box regression loss

                # Now, sum all the losses
                total_loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg

                # Perform backward pass with the total loss
                optimizer.zero_grad()
                total_loss.backward()  # Backpropagate with the total loss
                optimizer.step()  # Update model parameters
            
            avg_loss = epoch_loss/len(loader)
            train_losses.append(avg_loss)

            # Validation phase
            pretrained_model.eval()
            val_loss = 0.0
            val_map = 0.0
            val_map050 = 0.0
            metric = MeanAveragePrecision()
            metric.to(device)

            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    # 1. Get predictions (used for metric)
                    outputs = pretrained_model(images)
                    outputs = [filter_predictions(o) for o in outputs]
                    metric.update(outputs, targets)

                    # 2. Compute validation loss
                    loss_dict = pretrained_model(images, targets)  # This returns a dict of loss tensors
                    batch_loss = sum(
                        loss for loss in loss_dict
                        if isinstance(loss, torch.Tensor) and torch.isfinite(loss).all() and loss.numel() > 0
                    )
                    val_loss += batch_loss

                    # Compute metrics for mAP
                results = metric.compute()
                val_map += results["map"]  # mAP value
                val_map050 += results["map_50"]

            # Average the results over validation batches
            avg_val_loss = val_loss / len(val_loader)
            avg_val_map = val_map / len(val_loader)
            avg_val_map050 = val_map050 / len(val_loader)

            # Store validation results
            val_losses.append(avg_val_loss)
            val_mAP.append(avg_val_map)
            val_mAP050.append(avg_val_map050)

            # Print the results
            print(f"Epoch {epoch} - Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}, mAP@.5:.95: {results['map'].item():.4f}, mAP@.5: {results['map_50'].item():.4f}")


        print("Dragoooo! Dragooooooo! Dragoooooooooooo!")

        plt.figure()
        plt.plot(range(1, num_epochs+1), train_losses)
        plt.title("Training Loss Curve")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig("resnet_Loss_curve.png")

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs+1), val_mAP, label='Validation mAP@050:0.95')
        plt.xlabel('Epochs')
        plt.ylabel('mAP@050:0.95')
        plt.grid(True)
        plt.title('Validation mAP@050:0.95')
        plt.savefig("resnet_val_mAP@050:0.95_curve.png")

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs+1), val_mAP050, label='Validation mAP@0.50')
        plt.xlabel('Epochs')
        plt.ylabel('mAP@0.50')
        plt.grid(True)
        plt.title('Validation mAP@0.50')
        plt.savefig("resnet_val_mAP@0.50_curve.png")

        
    pretrained_model.eval()
    metric = MeanAveragePrecision()
    metric.to(device)

    with torch.no_grad():
        counter = 0

        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = pretrained_model(images)
            outputs = [filter_predictions(o) for o in outputs]
            # print(f"Predictions in batch: {[len(o['boxes']) for o in outputs]}")
            # for output in outputs:
            #     print("Scores:", output['scores'].cpu().numpy())

            metric.update(outputs, targets)
            if counter ==3:
                for output in pretrained_model(images[:2]):
                    print(output['labels'], output['scores'])
            if counter == 0:
                image_np = images[0].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
                orig_h, orig_w = targets[0]["orig_size"].cpu().numpy()
                image_np = np.clip(image_np, 0, 1)  # For float images from ToTensorV2
                image_np_resized = cv2.resize(image_np, (orig_w, orig_h))

                pred_boxes = outputs[0]['boxes'].cpu().numpy()
                pred_labels = outputs[0]['labels'].cpu().numpy()
                pred_scores = outputs[0]['scores'].cpu().numpy()

                h, w, _ = image_np_resized.shape  # Use original image dimensions

                # Create YOLO-like format expected by plot_image
                formatted_boxes = []
                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    x_center = x_min + width / 2
                    y_center = y_min + height / 2
                    formatted_boxes.append([label, score, x_center / w, y_center / h, width / w, height / h])  # normalized to original size

                plot_image(image_np_resized, formatted_boxes, pred_labels, pred_scores, image_index=0)
            counter += 1

        pred_boxes = outputs[0]['boxes'].cpu()
        # filtered_preds = [filter_predictions(p) for p in pred_boxes]
        gt_boxes = targets[0]['boxes'].cpu()

        ious = box_iou(pred_boxes, gt_boxes)
        print("IoU Matrix:\n", ious)

        results = metric.compute()

    
    visualize_predictions(images[0].cpu(), outputs[0])

    images, targets = next(iter(val_loader))
    images = [img.to(device) for img in images]
    with torch.no_grad():
        outputs = pretrained_model(images)
        outputs = [filter_predictions(o) for o in outputs]

    visualize_preds_vs_gt(images[0].cpu(), outputs[0], targets[0])

    print("mAP at IoU=0.50:0.95: ", results['map'])
    print("mAP at IoU=0.50: ", results['map_50'])
    if 'precision' in results and results['precision'].numel() > 0:
        precision = results['precision'][0].mean().item()
        recall = results['recall'][0].mean().item()
        print(f"precision: {precision:.4f}, recall: {recall:.4f}")
    else:
        print("No true positives detected — precision/recall not available.")    
    if 'recall' in results and results['recall'].numel() > 0:
        print("recall: ", results['recall'][0].mean().item())
    else:
        print("Recall not available — likely no matched predictions.")
