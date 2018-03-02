import argparse
import os
import coco
import model as modellib
import skimage.io
from skimage.measure import find_contours
import numpy as np
import json

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: CLASS_NAMES.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

parser = argparse.ArgumentParser()
parser.add_argument('image_file')
parser.add_argument('--pretrained_weights', default='mask_rcnn_coco.h5')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
if args.verbose:
    config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='.', config=config)

# Load weights trained on MS-COCO
model.load_weights(args.pretrained_weights, by_name=True)

image = skimage.io.imread(args.image_file)

# Run detection
results = model.detect([image], verbose=int(args.verbose))

# Visualize results
r = results[0]
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], CLASS_NAMES, r['scores'])

outputs = []
masks = np.moveaxis(r['masks'], 2, 0)
for roi, mask, class_id, score in zip(r['rois'], masks, r['class_ids'], r['scores']):
    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    # Subtract the padding and flip (y, x) to (x, y)
    contours = np.array([np.fliplr(verts) - 1 for verts in contours])
    
    output = {
        'roi': roi.tolist(),
        #'mask': mask.tolist(),
        'contours': contours.tolist(),
        'class_id': int(class_id),
        'class': CLASS_NAMES[class_id],
        'score': float(score)
    }
    outputs.append(output)

print(json.dumps(outputs))
