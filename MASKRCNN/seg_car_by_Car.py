import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from samples/coco/ import coco
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import car

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNNc
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
car_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_car_1000_50_0.h5")

# print(car_MODEL_PATH)
# Download COCO trained weights from Releases if needed
if not os.path.exists(car_MODEL_PATH):
    utils.download_trained_weights(car_MODEL_PATH)
# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
TEST_IMAGE_DIR = os.path.join(ROOT_DIR, "test_images")

class InferenceConfig(car.CarConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(car_MODEL_PATH, by_name=True)

# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')

class_names = ['BG', 'car']
			   
file_names = next(os.walk(TEST_IMAGE_DIR))[2]
N = len(file_names)

for i in range(N):
    # Load a random image from the images folder
    image = skimage.io.imread(os.path.join(TEST_IMAGE_DIR, file_names[i]))
    # Run detection
    results = model.detect([image], verbose=1)
    # Visualize results
    r = results[0]
    visualize.display_instances(image, file_names[i],r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])