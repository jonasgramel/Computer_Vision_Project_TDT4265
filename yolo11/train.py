from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Define which layers to unfreeze at which epochs
unfreeze_plan = {
    2: [9],              # at epoch 2, unfreeze model.9 SPFF
    5: [5, 6, 7, 8],     # at epoch 5, unfreeze model.5-8 Conv and C3k2
    8: [0, 1, 2, 3, 4],  # at epoch 8, unfreeze model.0-4 Conv and C3k2
}

# Custom callback to unfreeze at a certain epoch
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

# Initially freeze model.0 to model.9
for name, param in model.model.named_parameters():
    if any(f"model.{i}" in name for i in range(10)):
        param.requires_grad = False

# Train with automatic unfreezing
model.train(
    data="/home/audunsor/Documents/Computer_Vision_Project_TDT4265/yolo11/data.yaml",
    epochs=400,
    imgsz=1024,
    project="yolo11",
    name="yolo11s_lidar_transfer_learning_1024imgsz_400epochs",
    patience=50
)

model.predict(
    source="/work/datasets/tdt4265/ad/open/Poles/lidar/combined_color/test",
    project="yolo11",
    name="yolo11s_lidar_predictions_1024imgsz_400epochs",
    save_txt=True,
    save_conf=True # <--- This adds the probability of each predicted box
)