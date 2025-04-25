import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_prediction(image_tensor, pred, target, score_threshold=0.25):
    image = image_tensor.cpu().permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    # Draw predicted boxes
    boxes = pred["boxes"].cpu()
    scores = pred["scores"].cpu()
    labels = pred["labels"].cpu()

    for box, score in zip(boxes, scores):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{score:.2f}", color='lime', fontsize=10, backgroundcolor='black')

    # Draw ground truth boxes
    for gt_box in target["boxes"].cpu():
        x1, y1, x2, y2 = gt_box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

    plt.title("Green = Prediction, Red = Ground Truth")
    plt.axis("off")
    plt.tight_layout()
    plt.show()