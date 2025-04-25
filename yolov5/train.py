from tools.model_structure import model, scaled_anchors
from tools.file_reading import Dataset, collate_fn#load_images_and_labels, 
from tools.transforms import train_transform, test_transform
import torch
import torch.nn as nn
# from torchvision.ops import nms, box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tools.model_structure import YOLOLoss
from tqdm import tqdm


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
   
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # print(f"Input image shape: {images.shape}")

        raw_outputs = model.model(images)

        loss = loss_fn(raw_outputs, targets, scaled_anchors)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_description(f"Loss: {loss.item():.4f}")
    
    return losses   

if __name__ == "__main__": 
    # Load the model
    model = model.to(device)
    # Initialize the loss function
    loss_fn = YOLOLoss(anchors=scaled_anchors)
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(max_epochs):
        losses = training_loop(model, train_loader, optimizer, loss_fn, scaled_anchors)
        print(f"Epoch {epoch+1}/{max_epochs}, Loss: {sum(losses)/len(losses):.4f}")