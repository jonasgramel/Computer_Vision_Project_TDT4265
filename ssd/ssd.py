import torch
from torchvision.models.detection import ssd300_vgg16
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from torchvision.models.detection.ssd import SSDClassificationHead

from utils.box_preparation import convert_cells_to_bboxes
from utils.file_reading import Dataset
from utils.metrics_calculation import mean_average_precision
from utils.plot_picture import plot_image, visualize_dataset_sample, visualize_predictions

import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 
import matplotlib.pyplot as plt
import numpy as np
np.float = float


num_classes = 2
in_channels=[512, 1024, 512, 256, 256, 256]
num_anchors=[4, 6, 6, 6, 4, 4]

device = torch.device('cuda')
# Learning rate for training 
learning_rate = 5e-5

# Number of epochs for training 
num_epochs = 10

# Image size 
image_size = 300

batch_size=32

pretrained_model = ssd300_vgg16(weights="COCO_V1")

# Freeze the backbone (VGG16 part)
for param in pretrained_model.backbone.parameters():
    param.requires_grad = True

# Freeze all layers in the SSD head except for the final classifier
for param in pretrained_model.head.parameters():
    param.requires_grad = True

for param in pretrained_model.head.classification_head.parameters():
    param.requires_grad = True  # Only train the classification head

pretrained_model.head.classification_head = SSDClassificationHead(
    in_channels=in_channels,
    num_anchors=num_anchors,
    num_classes=num_classes
)

optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, pretrained_model.parameters()), 
    lr=0.001, momentum=0.9, weight_decay=0.0005
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


test_transform = A.Compose( 
	[ 
		# Rescale an image so that maximum side is equal to image_size 
		A.LongestMaxSize(max_size=image_size), 
		# Pad remaining areas with zeros 
		A.PadIfNeeded( 
			min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
		), 
		# Normalize the image 
		A.Normalize( 
			mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
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
    transform=test_transform
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=collate_fn
)

def filter_predictions(pred, score_thresh=0.5):
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

trainssd = True

if __name__ == "__main__":
    
    pretrained_model.to(device)

    if trainssd:
        
        print("♫Training Montage♫ by Vince DiCola starts playing...")
        train_losses = []
        pretrained_model.train()

        for i in range(3):
            visualize_dataset_sample(dataset, idx=i)

        for epoch in tqdm.trange(num_epochs):
            epoch_loss = 0
            n_batches = 0

            for images, labels in loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in labels]  # 'labels' is the correct name here

                for t in targets:
                    #print("labels:", t['labels'])
                    #print("boxes:", t['boxes'])

                    if not torch.isfinite(t['boxes']).all():
                        print("invalid box detected:", t['boxes'])

                    if not torch.isfinite(t['labels']).all():
                        print("invalid label detected:", t['labels'])             


                loss_dict = pretrained_model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()
                n_batches += 1
            
            avg_loss = epoch_loss/n_batches
            train_losses.append(avg_loss)

            print(f"Epoch {epoch} - Loss: {losses.item():.4f}")
        print("Dragoooo! Dragooooooo! Dragoooooooooooo!")

        plt.figure()
        plt.plot(range(1, num_epochs+1), train_losses, marker='o')
        plt.title("Training Loss Curve")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig("ssd_Loss_curve.png")
        plt.show()

        
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
        # filtered_preds = [filter_predictions(p, score_thresh=0.5) for p in pred_boxes]
        gt_boxes = targets[0]['boxes'].cpu()

        ious = box_iou(pred_boxes, gt_boxes)
        print("IoU Matrix:\n", ious)

        results = metric.compute()

    
    visualize_predictions(images[0].cpu(), outputs[0])

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
