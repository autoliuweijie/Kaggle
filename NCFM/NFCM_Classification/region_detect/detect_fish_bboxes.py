import os, sys
import numpy as np
import json
from mylib.models import Ipv3BboxClassifier, Ipv3BboxRegressor
from mylib.models import FishDetector
from mylib.dataio import TestDataReader


file_dir = os.path.dirname(os.path.abspath(__file__))


# settings
seed = 7
np.random.seed(seed)
debug = 0
ipv3_bbox_classifier_weights = os.path.join(
    file_dir, 
    'ipv3_bbox_classifier_weights/ipv3_bbox_classifier-03--0.00--1.00.h5',
)
input_shape = (299, 299, 3)
ipv3_bbox_regressor_weights = os.path.join(
    file_dir,
    'ipv3_bbox_regressor_weights_2/ipv3_bbox_regressor_nb_5-184--10.47-19.39.h5',
)
nb_bbox = 5


# load test image
test_data_reader = TestDataReader()
images_test, image_names_test, _ = test_data_reader.load_test_image_bbox_imgname(debug=debug)


# create Fish Detector
bbox_regressor = Ipv3BboxRegressor(weights_file=ipv3_bbox_regressor_weights, nb_bbox=nb_bbox)  # create bbox regressor
bbox_classifier = Ipv3BboxClassifier(weights_file=ipv3_bbox_classifier_weights, input_shape=input_shape)  # create bbox classifier
fish_detector = FishDetector(bbox_regressor, bbox_classifier)


# detect
fish_bboxes_test = fish_detector.detect(images_test, thresh=0.8, verbose=20)


# json serialize
json_dict = {}
for i in range(len(images_test)):
    image_name = image_names_test[i]
    bboxes = fish_bboxes_test[i]
    json_dict[image_name] = bboxes

str_json = json.dumps(json_dict)
with open("./outputs/my_test_stg1.json", 'w') as f:
    f.write(str_json)
print("Bbox has been saved at ./outputs/my_test_stg1.json!")
