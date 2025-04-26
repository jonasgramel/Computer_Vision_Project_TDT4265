from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="/home/audunsor/Documents/Computer_Vision_Project_TDT4265/yolo11/data.yaml",
    epochs=50,
    imgsz=640,
    lr0=0.001,
    pretrained=True,
    freeze=10,
    batch=32,
    project="yolo11",
    name="transfer_learning"
)

model.predict(
    source="/work/datasets/tdt4265/ad/open/Poles/rgb/images/test",
    project="yolo11",
    name="predictions",
    save_txt=True,
    save_conf=True # <--- This adds the probability of each predicted box
    )