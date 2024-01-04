label_list = [
    "airport", "animal", "beach", "bear", "bird", "boat", "book", "bridge", 
    "building", "car", "castle", "cat", "cityscape", "cloud", "computer", 
    "coral", "cow", "dancing", "dog", "earthquake", "elk", "fire", "fish", 
    "flag", "flower", "food", "fox", "frost", "garden", "glacier", "grass", 
    "harbor", "horse", "house", "lake", "leaf", "map", "military", "moon", 
    "mountain", "nighttime", "ocean", "person", "plane", "plant", "police", 
    "protest", "railroad", "rainbow", "reflection", "road", "rock", "running", 
    "sand", "sign", "sky", "snow", "soccer", "sport", "statue", "street", 
    "sun", "sunset", "surf", "swimmer", "tattoo", "temple", "tiger", "tower", 
    "town", "toy", "train", "tree", "valley", "vehicle", "water", "waterfall", 
    "wedding", "whale", "window", "zebra", "couch",
    "sheep", "bicycle", "bus", "motorcycle", "bottle", "chair", "diningtable", "pottedplant", 
    "sofa", "tvmonitor", "traffic light", "fire hydrant", "stop sign", "parking meter", 
    "bench", "giraffe", "backpack", "umbrella", "handbag", "truck", "elephant",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
    "baseball glove", "skateboard", "surfboard", "tennis racket", "wine glass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", 
    "carrot", "hot dog", "pizza", "donut", "cake", "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "airplane"
]


# label_list = [
#     "person", "bird", "cat", "cow", "dog", "horse", "sheep", "bicycle", "boat", "bus", 
#     "car", "motorcycle", "train", "bottle", "chair", "diningtable", "pottedplant", 
#     "sofa", "tvmonitor", "traffic light", "fire hydrant", "stop sign", "parking meter", 
#     "bench", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "truck", "elephant",
#     "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
#     "baseball glove", "skateboard", "surfboard", "tennis racket", "wine glass", "cup", 
#     "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", 
#     "carrot", "hot dog", "pizza", "donut", "cake", "bed", "toilet", "laptop", "mouse",
#     "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
#     "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "airplane"
# ]

id2label = {i : label_list[i] for i in range(len(label_list))}
label2id = {k : v for v, k in id2label.items()}
list_length = len(label_list)

pascal_labels = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable','dog', 'horse', 'motorcycle', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor']

coco_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
               'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
               'donut', 'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable', 'toilet', 
               'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
               'hair drier', 'toothbrush']

nuswide_labels = ["airport", "animal", "beach", "bear", "birds", "boats", "book", "bridge", 
                  "buildings", "cars", "castle", "cat", "cityscape", "clouds", "computer", 
                  "coral", "cow", "dancing", "dog", "earthquake", "elk", "fire", "fish", 
                  "flags", "flowers", "food", "fox", "frost", "garden", "glacier", "grass",
                  "harbor", "horses", "house", "lake", "leaf", "map", "military", "moon", 
                  "mountain", "nighttime", "ocean", "person", "plane", "plants", "police", 
                  "protest", "railroad", "rainbow", "reflection", "road", "rocks", "running", 
                  "sand", "sign", "sky", "snow", "soccer", "sports", "statue", "street", 
                  "sun", "sunset", "surf", "swimmers", "tattoo", "temple", "tiger", "tower", 
                  "town", "toy", "train", "tree", "valley", "vehicle", "water", "waterfall", 
                  "wedding", "whales", "window", "zebra"]