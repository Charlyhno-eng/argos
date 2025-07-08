from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=40,
    imgsz=416,
    batch=8,
    name="humans",
    device="cpu",
    workers=2,
    patience=10,
    save=True,
    val=True
)
