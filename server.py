import argparse
import tensorflow as tf
import keras.backend as K
import coco
import model as modellib
from flask import Flask, request, jsonify
import skimage
import skimage.measure
import numpy as np

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
parser.add_argument('--host', default='0.0.0.0')
parser.add_argument('--port', type=int, default=5000)
parser.add_argument('--pretrained_weights', default='mask_rcnn_coco.h5')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# to allow several web processes to share GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

config = InferenceConfig()
if args.verbose:
    config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='.', config=config)

# Load weights trained on MS-COCO
model.load_weights(args.pretrained_weights, by_name=True)

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route("/detect", methods=['POST'])
def detect():
    assert 'image' in request.files
    image_file = request.files['image']
    image = skimage.io.imread(image_file)

    # Run detection
    results = model.detect([image], verbose=int(args.verbose))
    r = results[0]

    outputs = []
    masks = np.moveaxis(r['masks'], 2, 0)
    for roi, mask, class_id, score in zip(r['rois'], masks, r['class_ids'], r['scores']):
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = skimage.measure.find_contours(padded_mask, 0.5)
        # Subtract the padding and flip (y, x) to (x, y)
        contours = [(np.fliplr(verts) - 1).tolist() for verts in contours]
        
        output = {
            'roi': roi.tolist(),
            #'mask': mask.tolist(),
            'contours': contours,
            'class_id': int(class_id),
            'class': CLASS_NAMES[class_id],
            'score': float(score)
        }
        outputs.append(output)

    return jsonify(outputs)

if __name__ == '__main__':
    app.run(args.host, args.port)
