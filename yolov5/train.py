from tools.model_structure import model, scaled_anchors
from tools.file_reading import Dataset, collate_fn#load_images_and_labels, 
from tools.transforms import train_transform, test_transform
import torch
import torch.nn as nn
from tools.visualize import visualize_prediction

# from torchvision.ops import nms, box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tools.model_structure import YOLOLoss
from tqdm import tqdm
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

logger = 'TensorBoard'
max_epochs = 10
batch_size = 16

train_dataset = Dataset( 
			# csv_file="./data/pascal voc/train.csv", 
			image_dir = "/work/datasets/tdt4265/ad/open/Poles/lidar/combined_color/train", # For Cybele, lidar images
			label_dir = "/work/datasets/tdt4265/ad/open/Poles/lidar/labels/train", # For Cybele, lidar labels
			transform=train_transform 
		) 

		# Defining the train data loader 
train_loader = torch.utils.data.DataLoader( 
    train_dataset, 
    batch_size = batch_size, 
    num_workers = 2, 
    shuffle = True, 
    pin_memory = True, 
    collate_fn=collate_fn,
)

val_dataset = Dataset( 
			# csv_file="./data/pascal voc/train.csv", 
			image_dir = "/work/datasets/tdt4265/ad/open/Poles/lidar/combined_color/valid", # For Cybele, lidar images
			label_dir = "/work/datasets/tdt4265/ad/open/Poles/lidar/labels/valid", # For Cybele, lidar labels
			transform=test_transform 
		) 

		# Defining the train data loader 
val_loader = torch.utils.data.DataLoader( 
    val_dataset, 
    batch_size = batch_size, 
    num_workers = 2, 
    shuffle = True, 
    pin_memory = True,
    collate_fn=collate_fn, 
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def training_loop(model, train_loader, optimizer, loss_fn, scaled_anchors):
    model.train()
    progress_bar = tqdm(train_loader, leave=True) 
    losses = []
    for batch_idx, (images, targets) in enumerate(progress_bar):
   
        images = torch.stack(images).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # print(f"Input image shape: {images.shape}")

        raw_outputs = model(images)

        loss = loss_fn(raw_outputs, targets, scaled_anchors)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_description(f"Loss: {loss.item():.4f}")
    
    return losses   

@torch.no_grad()
def validation_loop(model, val_loader, device):
    model.eval()
    metrics = MeanAveragePrecision()

    for images, targets in tqdm(val_loader, desc="Validating"):
        images = list(images)
        targets = list(targets)

        for i, (image, target) in enumerate(zip(images, targets)):
            image = image.to(device).unsqueeze(0)
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            target = {k: v.to(device) for k, v in target.items()}

            output = model(image)[0]   # Inference output
            output = output[0]         # remove batch dim
            
            if output.numel() == 0:
                pred = {
                    'boxes': torch.zeros((0, 4), device=device),
                    'scores': torch.zeros((0,), device=device),
                    'labels': torch.zeros((0,), dtype=torch.int64, device=device)
                }
            else:
                pred = {
                    'boxes': output[:, :4],
                    'scores': output[:, 4],
                    'labels': output[:, 5].long() if output.shape[1] > 5 else torch.zeros_like(output[:, 4]).long()
                }
            
            if i < 3:
                visualize_prediction(image.squeeze(0), pred, target)
        # Update metrics
            metrics.update([pred], [target])
    
    results = metrics.compute()
    print(f"Validation mAP: {results['map']:.4f}")

    for k,v in results.items():
        print(f"{k}: {v}")
    
    return results


if __name__ == "__main__": 
    # Load the model
    model = model.to(device)
    # Initialize the loss function
    loss_fn = YOLOLoss(anchors=scaled_anchors)
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_arr = []
    for epoch in range(max_epochs):
        losses = training_loop(model, train_loader, optimizer, loss_fn, scaled_anchors)
        loss_arr.append(losses)
        print(f"Epoch {epoch+1}/{max_epochs}, Loss: {sum(losses)/len(losses):.4f}")

        # Validation
        if epoch % 5 == 0:
            validation_loop(model, val_loader, device)

    plt.figure(figsize=(10, 5))
    plt.plot(loss_arr)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("training_loss.png")

    