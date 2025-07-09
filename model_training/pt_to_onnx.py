from ultralytics import YOLO

# Load the YOLO8 model
model = YOLO("human_detection.pt")

# Export the model to ONNX format
model.export(format="onnx", imgsz=(416, 416))
