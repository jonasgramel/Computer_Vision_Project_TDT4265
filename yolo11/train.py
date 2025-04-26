from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(data="data.yaml", epochs=100, imgsz=640)
model.predict(
    source="/work/datasets/tdt4265/ad/open/Poles/rgb/images/test",
    project="yolo11",
    name="predictions",
    save_txt=True,
    save_conf=True # <--- This adds the probability of each predicted box
    )