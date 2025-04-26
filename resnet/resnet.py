import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.models as models
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from torchvision.ops import box_area 

from utils.box_preparation import convert_cells_to_bboxes
from utils.file_reading import Dataset
from utils.metrics_calculation import iou_loss, compute_iou, FocalLoss
from utils.plot_picture import plot_image, visualize_dataset_sample, visualize_predictions, visualize_preds_vs_gt

import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 
import os
import matplotlib.pyplot as plt
import numpy as np
np.float = float


num_classes = 2

device = torch.device('cuda')
# Learning rate for training 
learning_rate = 1e-5

# Number of epochs for training 
num_epochs = 100

# Image size 
image_size = 224

batch_size=16


anchor_generator = AnchorGenerator(
    sizes=((32,), (64,), (128,), (256,), (512,)),
    aspect_ratios=((0.05, 0.1, 0.2),) * 5  # Modify aspect ratios for tall objects
)

# Define ResNet50 backbone with FPN
backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(backbone_name='resnet50', pretrained=True)

pretrained_model = FasterRCNN(
    backbone=backbone,
    num_classes=num_classes,
    rpn_anchor_generator=anchor_generator,
    rpn_pre_nms_top_n_train=4000,
    rpn_post_nms_top_n_train=2000,
    rpn_pre_nms_top_n_test=2000,
    rpn_post_nms_top_n_test=500,
    rpn_nms_thresh=0.4  # tighter threshold
)

pretrained_model = pretrained_model.to(device)

for name, param in pretrained_model.backbone.body.named_parameters():
    if "conv1" in name or "bn1" in name:
        param.requires_grad = False  # Freeze very early layers (edges/textures)
    else:
        param.requires_grad = True  # Allow all ResNet blocks to adapt

in_features = pretrained_model.roi_heads.box_predictor.cls_score.in_features
pretrained_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, pretrained_model.parameters()),
    lr=1e-5, weight_decay=0.001  # Slightly stronger weight decay
)

train_transform = A.Compose([
    A.LongestMaxSize(max_size=image_size),
    A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
    
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.2),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.ShiftScaleRotate(
        shift_limit=0.02, scale_limit=(0.1, 0.2), rotate_limit=5, p=0.5, border_mode=cv2.BORDER_CONSTANT
    ),
    A.RandomSizedBBoxSafeCrop(height=image_size, width=image_size, erosion_rate=0.2, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2, label_fields=[]))


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

def predict(model, image_size, output_folder, device='cuda'):
    """
    Predict bounding boxes using a PyTorch model on the test set.

    Args:
        model: The trained PyTorch model.
        image_size: The size of the image to resize to (e.g., 224).
        output_folder: Folder where the predictions will be saved.
        device: Device to run the model on ('cuda' or 'cpu').
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
    
    # Define the transformation for the test set (same as the val_transform)
    test_transform = A.Compose(
        [
            # Rescale the image so that the maximum side is equal to image_size
            A.LongestMaxSize(max_size=image_size),
            # Pad the remaining areas with zeros (to ensure the final image size)
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
        ]
    )

    # Initialize the dataset for the test set
    test_dataset = Dataset(
        image_dir="/work/datasets/tdt4265/ad/open/Poles/lidar/combined_color/test",  # Use the "test" folder
        label_dir=None,  # Use the "test" folder
        transform=test_transform
    )

    # Create the DataLoader for the test set
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=1,  # Make sure to adjust this depending on your GPU memory
        shuffle=False,  # Do not shuffle in test set
        collate_fn=collate_fn  # Ensure that you have a collate_fn to handle different sizes of bounding boxes
    )

    # Iterate over the test set
    for images, targets, image_paths in test_loader:
        images = list(img.to(device) for img in images)  # Move images to device
        
        with torch.no_grad():  # No gradients are needed for inference
            predictions = pretrained_model(images)  # Run inference on the images

        # Assuming batch size is 1 (image_paths is a list with single path)
        for idx, prediction in enumerate(predictions):
            # Ensure we are accessing the image path correctly from the test loader
            image_path = image_paths[idx]  # This should be a string representing the image file path
            image_name = os.path.basename(image_path)  # Extract the file name (e.g., "image1.png")
            
            # Check if predictions have bounding boxes, labels, and scores
            if len(prediction['boxes']) == 0:  # No predictions
                print(f"No predictions for {image_name}")
                continue  # Skip to the next image
            
            boxes = prediction['boxes'].cpu().numpy()  # Bounding boxes
            labels = prediction['labels'].cpu().numpy()  # Class labels
            scores = prediction['scores'].cpu().numpy()  # Confidence scores

            # Define the text file path for saving predictions
            txt_filepath = os.path.join(output_folder, image_name.replace(".png", ".txt"))

            with open(txt_filepath, 'w') as f:
                for i in range(len(boxes)):
                    # YOLO format such that the boxes are in normalized coordinates
                    x_min, y_min, x_max, y_max = boxes[i]
                    
                    x_center = (x_min + x_max) / 2 / image_size
                    y_center = (y_min + y_max) / 2 / image_size
                    width = (x_max - x_min) / image_size
                    height = (y_max - y_min) / image_size
                    confidence = scores[i]
                    
                    # Write in YOLO format: class_id x_center y_center width height confidence
                    # Assuming the class_id is always 0 (as per your example)
                    f.write(f"0 {x_center} {y_center} {width} {height} {confidence}\n")

    print("Inference complete and results saved!")

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

def filter_predictions(pred, score_thresh=0.01):
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
        val_losses = []  # To track validation losses
        val_mAP = []  # To track validation mAP
        val_mAP050 = []  # To track validation mAP


        for i in range(3):
            visualize_dataset_sample(dataset, idx=i)

        for epoch in tqdm.trange(num_epochs):

            if epoch == 0:
                for g in optimizer.param_groups:
                    g['lr'] = 1e-4  # Start a bit higher for first epoch
            if epoch == 1:
                print("Unfreezing the whole backbone (except conv1/bn1)...")
                for name, param in pretrained_model.backbone.body.named_parameters():
                    if "conv1" not in name and "bn1" not in name:
                        param.requires_grad = True
            if epoch == 2:
                for g in optimizer.param_groups:
                    g['lr'] = 1e-5  # Then back to normal
                for name, param in pretrained_model.backbone.body.named_parameters():
                    if "conv1" in name or "bn1" in name:
                        param.requires_grad = True  # Freeze very early layers (edges/textures)
                    else:
                        param.requires_grad = True  # Allow all ResNet blocks to adapt
            # if epoch == 5:  # Unfreeze 'layer3'
            #     print("Unfreezing layer 3. Cool party!")
            #     torch.cuda.empty_cache()
            #     for g in optimizer.param_groups:
            #         g['lr'] = 5e-6
            #     for param in pretrained_model.backbone.body.layer3.parameters():
            #         param.requires_grad = True
            # elif epoch == 10:  # Unfreeze 'layer2'
            #     print("Unfreezing layer 2. What killed the dinosaurs? The Ice Age!")
            #     torch.cuda.empty_cache()
            #     for g in optimizer.param_groups:
            #         g['lr'] = 1e-6
            #     for param in pretrained_model.backbone.body.layer2.parameters():
            #         param.requires_grad = True
            # elif epoch == 15:  # Unfreeze the entire backbone
            #     torch.cuda.empty_cache()
            #     for g in optimizer.param_groups:
            #         g['lr'] = 1e-7
            #     print("Unfreezing the entire backbone. Everybody chill!")
            #     for param in pretrained_model.backbone.parameters():
            #         param.requires_grad = True
            
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
                epoch_loss += total_loss

                # Perform backward pass with the total loss
                optimizer.zero_grad()
                total_loss.backward()  # Backpropagate with the total loss
                torch.nn.utils.clip_grad_norm_(pretrained_model.parameters(), max_norm=1.0)
                optimizer.step()  # Update model parameters
            
            avg_loss = epoch_loss/len(loader)
            train_losses.append(avg_loss)

            # Validation phase
            pretrained_model.eval()
            val_epoch_loss = 0.0
            val_map = 0.0
            val_map050 = 0.0
            metric = MeanAveragePrecision(iou_thresholds=[0.3, 0.5, 0.75])
            metric.to(device)

            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    # Switch to training mode briefly to get loss dict
                    pretrained_model.train()
                    val_loss_dict = pretrained_model(images, targets)
                    pretrained_model.eval()  # switch back to eval

                    # Compute filtered predictions for mAP metric
                    with torch.no_grad():
                        outputs = pretrained_model(images)
                        outputs = [filter_predictions(o) for o in outputs]
                        metric.update(outputs, targets)

                    val_loss_classifier = val_loss_dict['loss_classifier']
                    val_loss_box_reg = val_loss_dict['loss_box_reg']
                    val_loss_objectness = val_loss_dict['loss_objectness']
                    val_loss_rpn_box_reg = val_loss_dict['loss_rpn_box_reg']

                    # Now, sum all the losses
                    val_total_loss = val_loss_classifier + val_loss_box_reg + val_loss_objectness + val_loss_rpn_box_reg
                    val_epoch_loss += val_total_loss

                    # Compute metrics for mAP
                results = metric.compute()
                val_map += results["map"]  # mAP value
                val_map050 += results["map_50"]

            # Average the results over validation batches
            avg_val_loss = val_epoch_loss / len(val_loader)
            avg_val_map = val_map / len(val_loader)
            avg_val_map050 = val_map050 / len(val_loader)

            # Store validation results
            val_losses.append(avg_val_loss)
            val_mAP.append(avg_val_map)
            val_mAP050.append(avg_val_map050)

            # Print the results
            print(f"Epoch {epoch} - Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}, mAP@.5:.95: {results['map'].item():.4f}, mAP@.5: {results['map_50'].item():.4f}")


        print("Dragoooo! Dragooooooo! Dragoooooooooooo!")

        train_losses_cpu = torch.tensor(train_losses).cpu().numpy()

        plt.figure()
        plt.plot(range(1, num_epochs+1), train_losses_cpu)
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
        plt.savefig("resnet_val_mAP@050-095_curve.png")

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs+1), val_mAP050, label='Validation mAP@0.50')
        plt.xlabel('Epochs')
        plt.ylabel('mAP@0.50')
        plt.grid(True)
        plt.title('Validation mAP@0.50')
        plt.savefig("resnet_val_mAP@050_curve.png")

        
    pretrained_model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.3, 0.5, 0.75])
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
        # print([box_area(b) for b in pred_boxes])
        # filtered_preds = [filter_predictions(p) for p in pred_boxes]
        gt_boxes = targets[0]['boxes'].cpu()

        ious = box_iou(pred_boxes, gt_boxes)
        print("IoU Matrix:\n", ious)

        results = metric.compute()

    
    visualize_predictions(images[0].cpu(), outputs[0])


    images, targets = next(iter(val_loader))
    images = [img.to(device) for img in images]

    with torch.no_grad():
        predictions = pretrained_model(images)  # Get raw predictions (not filtered yet)

        for idx, p in enumerate(predictions):
            boxes = p['boxes'].cpu().numpy()
            scores = p['scores'].cpu().numpy()
            labels = p['labels'].cpu().numpy()
            
            print(f"\n🔍 Image {idx}: {len(boxes)} boxes predicted")
            for score, box in zip(scores, boxes):
                print(f"Score: {score:.4f}, Box: {box}, Width: {box[2] - box[0]:.2f}, Height: {box[3] - box[1]:.2f}")

            # Optional: visualize low-score boxes too
            visualize_preds_vs_gt(images[idx].cpu(), p, targets[idx], idx=idx, type="lidar")

        # If you want to try filtering at 0.05:
        outputs = [filter_predictions(p, score_thresh=0.05) for p in predictions]
        visualize_preds_vs_gt(images[0].cpu(), outputs[0], targets[0], idx=3.14, type="lidar")

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



    predict(pretrained_model, image_size, "resnet/predictions", device='cuda')