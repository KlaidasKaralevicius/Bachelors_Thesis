from ultralytics import YOLO

model = YOLO("yolo26m.pt") # yolo26n-s-m-l-x.pt
# train models
train = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    optimizer='MuSGD',
    batch=0.80,
    seed=226,
    name='medium', # nano, small, medium, large, xlarge
    project='train',
    exist_ok=True,
    device=0
)