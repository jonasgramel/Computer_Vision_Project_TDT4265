import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead

from utils.box_preparation import convert_cells_to_bboxes
from utils.file_reading import Dataset
from utils.metrics_calculation import mean_average_precision
import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 
import numpy as np


num_classes = 2
device = torch.device('cuda')
# Learning rate for training 
learning_rate = 1e-5

# Number of epochs for training 
num_epochs = 5

# Image size 
image_size = 300

batch_size=8

pretrained_model = ssd300_vgg16(weights="COCO_V1")

# Freeze the backbone (VGG16 part)
for param in pretrained_model.backbone.parameters():
    param.requires_grad = False

# Freeze all layers in the SSD head except for the final classifier
for param in pretrained_model.head.parameters():
    param.requires_grad = True

for param in pretrained_model.head.classification_head.parameters():
    param.requires_grad = True  # Only train the classification head

pretrained_model.head.classification_head.num_classes = num_classes

optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, pretrained_model.parameters()), 
    lr=0.005, momentum=0.9, weight_decay=0.0005
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
					min_visibility=0.4, 
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
	transform=test_transform 
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


def convert_to_yolo_format(boxes, image_width, image_height):
    """ Convert bounding boxes to YOLO format [x_center, y_center, width, height] normalized by image size. """
    yolo_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        # Normalize by image size
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height
        yolo_boxes.append([x_center, y_center, width, height])
    return np.array(yolo_boxes)

def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Get predictions from the model
            predictions = model(images)
            
            for i, pred in enumerate(predictions):
                # Convert the predicted boxes to YOLO format
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                
                # Get ground truth boxes and labels
                true_boxes = targets[i]['boxes'].cpu().numpy()
                true_labels = targets[i]['labels'].cpu().numpy()

                # Convert ground truth boxes to YOLO format
                image_width = images[i].shape[2]
                image_height = images[i].shape[1]
                true_boxes_yolo = convert_to_yolo_format(true_boxes, image_width, image_height)
                pred_boxes_yolo = convert_to_yolo_format(pred_boxes, image_width, image_height)

                # Add the predictions and ground truth to the lists
                all_preds.append((pred_boxes_yolo, pred_labels, pred_scores))
                all_labels.append((true_boxes_yolo, true_labels))
    
    # Calculate mAP, precision, recall using your `mean_average_precision` function
    precisions, recalls, mAP = mean_average_precision(all_labels, all_preds)
    
    return mAP, precisions, recalls

images, targets = next(iter(loader))

trainssd = True
print("halla utenfor main")
if __name__ == "__main__":
    print("halla i main før train")
    if trainssd:
        print("halla")
        pretrained_model.to(device)
        pretrained_model.train()

        for epoch in tqdm.trange(num_epochs):
            for images, labels in loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in labels]  # 'labels' is the correct name here

                if any(len(t['boxes']) == 0 for t in targets):
                    continue
                loss_dict = pretrained_model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            print(f"Epoch {epoch} - Loss: {losses.item():.4f}")
        
    # Final evaluation after training
    mAP, precisions, recalls = evaluate_model(pretrained_model, val_loader, device)
    print(f"Final mAP: {mAP:.4f}")
    print(f"Final Precision: {precisions.mean():.4f}")
    print(f"Final Recall: {recalls.mean():.4f}")
    print("Final Evaluation")

    mAP, precision = mean_average_precision(pretrained_model, val_loader, device)
    print(f"Final mAP: {mAP:.4f}, Final Precision: {precision:.4f}")