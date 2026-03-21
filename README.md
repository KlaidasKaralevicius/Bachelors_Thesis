# Object Detection Using a Model Trained on Limited Data
### Description
Repository for Bachelor Thesis on how limited dataset contributes to custom solution object detection and accuracy vs performance balance launching object detection on edge device. Device - Raspberry Pi 5 4GB, Model - YOLO26 (all 'sizes': nano, small, medium...). Thesis purpose to test how YOLO works on custom data 'out of the box' and compare to same model after transfer-learning on limited dataset, then compare how accuracy is impacted (does limited dataset have impact, or diversity of data is hard requirement). After testing accuracy on PC with GPU, launch every model on edge device to see how different sizes impact performance (does extra X % of accuracy worth the Y % increase in latency) and can even specific model size be launched on limited resources.
### Files
- data/images - image folder;
- data/labels - YOLO style bounding box annoation files + classes.txt file to link class name to class index;
- data/predefined_classes.txt - all existing classes to create bounding boxes using labelImg tool;
- class_counter.py - python script to count how many times each object appears in dataset (after bounding box annotation);
- code/train.py - YOLO transfer-learning python script;
- code/test.py - launching YOLO inference to test accuracy between model sizes;
- raspberry_pi/pi_test.py - launch real-time inference on Raspberry Pi 5 to run object detection using YOLO model and save working metrics (latency, fps, CPU and RAM utilization, temperature, energy use);
- requirements.txt - pip packages to launch python scripts;
- raspberry_pi/metrics/ - text files with metrics from Raspberry Pi 5 runs for each YOLO size.
### Dataset
1050 images containg 14 classes, 1051 labels (1 for each image + .txt file linking class index to class name).\n
**Image description:**
- 20 empty;
- 302 multi-object;
- 52 single object for each class;
- Each class has 4 representations (different objects of same class, for example - different form and color cups);
- Fork – 135 total; 
- Spoon – 135 total; 
- Knife – 134 total; 
- Ladle – 134 total; 
- Spatula – 134 total; 
- Bowl – 134 total; 
- Plate – 135 total; 
- Cup – 134 total; 
- Frying pan – 134 total; 
- Sauce pan – 135 total; 
- Colander – 133 total; 
- Pan lid – 133 total; 
- Grater – 136 total; 
- Cutting board – 137 total.
**Image parameters:**
- 3 backgrounds;
- 20-90 angle;
- 15-55 cm distance from object;
- Blur - most images clear, some fully blurry, some slightly blurry;
- Random position in the image;
- Random obscuring of objects (part outside the image, behind/in front/on top/below other objects);
- 3 lighting: natural from side, artificial from top, camera flash;
- 1280x576 resolution;
- Photo taken - Samsung camera SM-A326B;
- 4.6 mm focal length;
- 1.80 EV aperture value;
- Automatic exposure time and ISO speed rating.
