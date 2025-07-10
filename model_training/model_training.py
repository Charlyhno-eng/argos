from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="/kaggle/working/data.yaml",
    epochs=40,                 # suffisant avec bon dataset
    imgsz=832,                 # réduit le temps vs 960 mais reste précis
    batch=8,                   # adapté au GPU de Kaggle (16 serait trop)
    name="humans_model",
    device=0,
    workers=2,
    save=True,
    val=True,
    augment=True,
    mosaic=0.2,               # plus faible = plus réaliste pour petits objets
    close_mosaic=10,          # désactive mosaic après 10 époques
    degrees=0.5,              # petites rotations
    scale=0.5,                # zooms modérés
    translate=0.1             # petits décalages
)
