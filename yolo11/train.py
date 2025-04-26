from ultralytics import YOLO

pretrained_model = YOLO("yolo11n.pt")
model = YOLO("yolo11n.yaml")

for new_layer, pre_layer in zip(model.model.model[:pretrained_model.model.model.yaml['backbone']], pretrained_model.model.model[:pretrained_model.model.model.yaml['backbone']]):
    new_layer.load_state_dict(pre_layer.state_dict())

results = model.train(
    data="/home/audunsor/Documents/Computer_Vision_Project_TDT4265/yolo11/data.yaml",
    epochs=50,
    imgsz=640,
    lr0=0.0005,
    pretrained=False,
    freeze=2,
    batch=32,
    project="yolo11",
    name="rgb_transfer_learning",
    hsv_h=0.015, # color jitter
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10, #rotation
    translate=0.1, #translation
    scale=0.5, #scaling
    shear=2.0,
    perspective=0.0005,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2
)

model.predict(
    source="/work/datasets/tdt4265/ad/open/Poles/rgb/images/test",
    project="yolo11",
    name="rgb_predictions",
    save_txt=True,
    save_conf=True # <--- This adds the probability of each predicted box
    )