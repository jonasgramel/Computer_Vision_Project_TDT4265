import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def plot_image(image, boxes, image_index): 
    """
    This function is gathered from Geeks for Geeks: https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/ 
    Accessed: 09-04-2025
    """
    # Getting the color map from matplotlib 
    colour_map = plt.get_cmap("tab20b") 
    colors = [colour_map(i) for i in np.linspace(0, 1, 20)] 
  
    # Reading the image with OpenCV 
    img = np.array(image) 
    # Getting the height and width of the image 
    h, w, _ = img.shape 
  
    # Create figure and axes 
    fig, ax = plt.subplots(1) 
  
    # Add image to plot 
    ax.imshow(img) 
  
    # Plotting the bounding boxes without labels
    for box in boxes: 
        class_pred = int(box[0])  # still used to color boxes
        box = box[2:] 
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
  
        rect = patches.Rectangle( 
            (upper_left_x * w, upper_left_y * h), 
            box[2] * w, 
            box[3] * h, 
            linewidth=2, 
            edgecolor=colors[class_pred % len(colors)], 
            facecolor="none", 
        ) 
          
        ax.add_patch(rect) 
  
    # Save the plot 
    plt.savefig("figures/figure" + str(image_index))