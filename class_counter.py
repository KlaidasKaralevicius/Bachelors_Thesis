import os
from collections import Counter

folder = "<Path to>/data/labels"

# load classes
with open(os.path.join(folder, "classes.txt"), "r") as f:
    classes = [line.strip() for line in f if line.strip()]

counts = Counter()
single_files = Counter()
empty_files = 0
multi_files = 0

for f in os.listdir(folder):
    if not f.endswith(".txt"):
        continue
    if f == "classes.txt":
        continue

    path = os.path.join(folder, f)

    file_classes = set()
    count = 0

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])

            counts[class_id] += 1
            file_classes.add(class_id)
            count += 1

    # empty file
    if count == 0:
        empty_files += 1
        continue

    # multi-object file
    if count > 1:
        multi_files += 1

    # single-object file
    if len(file_classes) == 1:
        only_class = next(iter(file_classes))
        single_files[only_class] += 1


# print counts
for i, name in enumerate(classes):
    total = counts[i]
    single = single_files[i]
    print(f"{name}: {total} total | {single} solo")

print(f"\nEmpty: {empty_files}")
print(f"Multi-object: {multi_files}")
