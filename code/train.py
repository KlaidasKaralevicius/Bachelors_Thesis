from ultralytics import YOLO

model = YOLO("./weights/yolo26x.pt") # yolo26n-s-m-l-x.pt
# train models
train = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    freeze=10,
    optimizer='MuSGD',
    batch=12,
    seed=226,
    name='xlarge', # nano, small, medium, large, xlarge
    project='train',
    exist_ok=True,
    device=0
)