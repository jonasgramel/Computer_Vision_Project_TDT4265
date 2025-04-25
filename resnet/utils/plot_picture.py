import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import torch


def plot_image(image, boxes, labels, scores, image_index, orig_h=None, orig_w=None): 
    """
    This function plots an image with bounding boxes and includes class labels and confidence scores.
    
    image: The input image (numpy array or tensor)
    boxes: List of bounding boxes in the format [class_id, x_center, y_center, width, height]
    labels: List of predicted labels corresponding to the boxes
    scores: List of predicted confidence scores corresponding to the boxes
    image_index: Index of the image in the batch (for saving filenames)
    orig_h: The original height of the image
    orig_w: The original width of the image (for scaling, if necessary)
    """
    # Getting the color map from matplotlib
    colour_map = plt.get_cmap("tab20b")
    colors = [colour_map(i) for i in np.linspace(0, 1, 20)]

    # Reading the image (if needed, convert tensor to numpy array)
    img = np.array(image)

    # Getting the height and width of the image
    h, w, _ = img.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Add image to plot
    ax.imshow(img)

    # Plotting the bounding boxes with labels and scores
    for box, label, score in zip(boxes, labels, scores):
        class_pred = int(label)  # Label is the predicted class
        box_coords = box[2:]  # Get the x_center, y_center, width, height
        upper_left_x = box_coords[0] - box_coords[2] / 2
        upper_left_y = box_coords[1] - box_coords[3] / 2

        # Create a rectangle for the bounding box
        rect = patches.Rectangle(
            (upper_left_x * w, upper_left_y * h), 
            box_coords[2] * w, 
            box_coords[3] * h, 
            linewidth=2, 
            edgecolor=colors[class_pred % len(colors)], 
            facecolor="none"
        )

        # Add rectangle to the plot
        ax.add_patch(rect)

        # Adding text (label and score) near the box
        ax.text(
            upper_left_x * w, 
            upper_left_y * h, 
            f'{label} ({score:.2f})', 
            color='white', fontsize=12, 
            verticalalignment='bottom', 
            bbox=dict(facecolor='red', alpha=0.5)
        )

    # Save the plot
    os.makedirs("figures", exist_ok=True)  # Ensure the directory exists
    fig_name = "resnet/figures/figure" + str(image_index) + ".png"
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close(fig)


def visualize_dataset_sample(dataset, idx=0, figsize=(6, 6)):
    image, target = dataset[idx]
    boxes = target['boxes']
    labels = target['labels']

    # Unnormalize image (ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean  # De-normalize
    image = torch.clamp(image, 0, 1)

    image_np = image.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image_np)

    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 2, f"Class {label.item()}", color='lime', fontsize=12, weight='bold')

    ax.set_title(f"Sample #{idx} — Boxes: {len(boxes)}")
    plt.axis('off')
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)  # Ensure the directory exists
    save_path = f"resnet/figures/train_pic_{idx}.png"
    plt.savefig(save_path)
    # plt.show()
    plt.close(fig)


def visualize_predictions(image, prediction, figsize=(6, 6), title="Predictions"):
    # Unnormalize image (ImageNet)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)

    image_np = image.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image_np)

    boxes = prediction['boxes'].cpu()
    scores = prediction['scores'].cpu()
    labels = prediction['labels'].cpu()

    for box, label, score in zip(boxes, labels, scores):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, f"Class {label.item()} ({score:.2f})",
                color='red', fontsize=10, weight='bold')

    ax.set_title(title)
    plt.axis('off')
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)  # Ensure the directory exists
    save_path = f"resnet/figures/prediction_pic.png"
    plt.savefig(save_path)
    # plt.show()
    plt.close(fig)


def visualize_preds_vs_gt(image, pred, gt, idx, image_size=300):

    # De-normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    image_np = image.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np)

    # Draw ground truth in green
    for box in gt['boxes'].cpu():
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

    # Draw predictions in red
    for box, score in zip(pred['boxes'].cpu(), pred['scores'].cpu()):
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='red', linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 4, f"{score:.2f}", color='red', fontsize=9)

    ax.set_title("Green: GT, Red: Predictions")
    plt.axis('off')
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    save_path = f"resnet/figures/pred_vs_gt_pic{idx}.png"
    plt.savefig(save_path)
    # plt.show()
    plt.close(fig)