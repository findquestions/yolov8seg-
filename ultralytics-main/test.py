from ultralytics import YOLO

model = YOLO("D:/Download/yolov8/ultralytics-main/weights/yolov8l-seg.pt")

results = model.train(
    batch=16,
    device="gpu",
    data="D:/Download/yolov8/ultralytics-main/ultralytics/cfg/myseg.yaml",
    epochs=10,
)

