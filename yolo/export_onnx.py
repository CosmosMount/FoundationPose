from ultralytics import YOLO

model = YOLO('detect/train2/weights/best.pt')
model.export(format='onnx')
