from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="pothole.yaml",
    epochs=1,
    imgsz=640,
    workers=0,
    cache=False
)