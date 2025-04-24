import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


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
    fig_name = "ssd/figures/figure" + str(image_index) + ".png"
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close(fig)