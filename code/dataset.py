import os
import random
import shutil
import json
from collections import defaultdict

data_dir = "../data"
out_dir = "../split_dataset"
img_dir = os.path.join(data_dir, "images")
label_dir = os.path.join(data_dir, "labels")
train_split = 0.75
val_split = 0.15
test_split = 0.10
random.seed(226)

# ===== LOAD DATA =====
image_objects = {}   # image: list of class ids
all_classes = set()

for file in os.listdir(label_dir):
    # skip class file 
    if not file.endswith(".txt") or file == "classes.txt":
        continue
    
    # read data of each label file
    path = os.path.join(label_dir, file)
    with open(path, "r") as f:
        lines = f.readlines()
    
    # collect all classes
    classes = []
    for line in lines:
        if line.strip() == "":
            continue
        cls = int(line.split()[0])
        classes.append(cls)
        all_classes.add(cls)
    
    # connect label file with image
    img_name = file.replace(".txt", ".jpg")
    image_objects[img_name] = classes

# randomize order
images = list(image_objects.keys())
random.shuffle(images)

# ===== SPLIT TEST ======
test_size = int(len(images) * test_split)
test_set = set(images[:test_size])
remaining = images[test_size:]

# ===== GREEDY BALANCE TRAIN/VAL =====
train_set = set()
val_set = set()
train_counts = defaultdict(int)
val_counts = defaultdict(int)
target_train = int(len(images) * train_split)
target_val = int(len(images) * val_split)

for img in remaining:
    classes = image_objects[img]
    # score how needed is the class
    train_score = sum(train_counts[c] for c in classes)
    val_score = sum(val_counts[c] for c in classes)
    
    # assign data untill it is full
    if len(train_set) >= target_train:
        chosen = "val"
    elif len(val_set) >= target_val:
        chosen = "train"
    else:
        chosen = "train" if train_score <= val_score else "val"
    
    # add data to split and monitor counts
    if chosen == "train":
        train_set.add(img)
        for c in classes:
            train_counts[c] += 1
    else:
        val_set.add(img)
        for c in classes:
            val_counts[c] += 1
            
# ===== YOLO STRUCTURE =====
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

# create folder structure
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

# copy images to new folder using split structure
def copy_split(split_set, split_name):
    for img in split_set:
        lbl = img.replace(".jpg", ".txt")
        
        shutil.copy(
            os.path.join(img_dir, img),
            os.path.join(out_dir, "images", split_name, img)
        )
        
        # handle empty images (no label file)
        lbl_path = os.path.join(label_dir, lbl)
        if os.path.exists(lbl_path):
            shutil.copy(
                lbl_path,
                os.path.join(out_dir, "labels", split_name, lbl)
            )
        else:
            # create empty label file
            open(os.path.join(out_dir, "labels", split_name, lbl), "w").close()

copy_split(train_set, "train")
copy_split(val_set, "val")
copy_split(test_set, "test")

# ===== SPLITS METADATA JSON =====
split_data = {
    "train": sorted(list(train_set)),
    "val": sorted(list(val_set)),
    "test": sorted(list(test_set))
}

with open(os.path.join(out_dir, "split.json"), "w") as f:
    json.dump(split_data, f, indent=2)
    
# ===== PRINT STATS =====
def compute_counts(split_set):
    counts = defaultdict(int)
    for img in split_set:
        for c in image_objects[img]:
            counts[c] += 1
    return counts

train_stats = compute_counts(train_set)
val_stats = compute_counts(val_set)
test_stats = compute_counts(test_set)

print("\nClass distribution:")
print("CLASS | TRAIN | VAL | TEST")

classes_path = os.path.join(label_dir, "classes.txt")

class_names = []
if os.path.exists(classes_path):
    with open(classes_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

for c in sorted(all_classes):
    name = class_names[c] if c < len(class_names) else str(c)
    print(f"{c:5} ({name:10}) | {train_stats[c]:5} | {val_stats[c]:5} | {test_stats[c]:5}")

print(len(train_set), len(val_set), len(test_set))
print(len(train_set) + len(val_set) + len(test_set))