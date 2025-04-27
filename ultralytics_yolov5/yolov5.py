from ultralytics import YOLO

model = YOLO("yolov5s.pt")  # Load a pretrained model

image_type = "rgb"
# image_type = "lidar"
max_epochs = 400
image_size = 1024


unfreeze_plan = {
    2: [9],              # unfreeze SPPF at epoch 2
    5: [5, 6, 7, 8],     # unfreeze deeper backbone (conv and small C3 layers) at epoch 5
    8: [0, 1, 2, 3, 4],  # unfreeze early feature extractor at epoch 8
}

def progressive_unfreeze(trainer):
    current_epoch = trainer.epoch
    if current_epoch in unfreeze_plan:
        layers_to_unfreeze = unfreeze_plan[current_epoch]
        print(f">>> Unfreezing layers {layers_to_unfreeze} at epoch {current_epoch}")
        for name, param in trainer.model.model.named_parameters():
            if any(f"model.{i}" in name for i in layers_to_unfreeze):
                param.requires_grad = True
    
# Register the callback
model.add_callback("on_train_epoch_start", progressive_unfreeze)

if image_type == "rgb":
    data_file = "rgb_data.yaml"
    test_source = "/work/datasets/tdt4265/ad/open/Poles/rgb/images/test"
    project_name = "runs/rgb" + str(max_epochs)

elif image_type == "lidar":
    data_file = "lidar_data.yaml"
    test_source = "/home/jonasgm/Documents/TDT4265/Mini_project/lidar/images/test"
    project_name = "runs/lidar"+ str(max_epochs)

else:
    raise ValueError("Invalid image type. Choose 'rgb' or 'lidar'.")

# Define which layers to unfreeze at which epochs

model.train(
    data=data_file, 
    epochs=max_epochs, 
    imgsz=image_size,
    project=project_name,
    freeze=10,
    patience=50)  # Train the model


model.predict(
    source = test_source,
    project=project_name,
    name="predictions",
    save_txt=True,
    save_conf=True # This adds the probability of each predicted box
    )

