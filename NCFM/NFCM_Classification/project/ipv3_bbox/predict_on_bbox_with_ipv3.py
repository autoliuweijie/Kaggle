import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
import numpy as np
from models.classifiers import Ipv3BboxImageClassifier
from myutils.dataio import TestDataReader, make_submission, limit_upper_and_lower, translate_scores_to_onehot
import matplotlib.pyplot as plt


# settings
seed = 7
np.random.seed(seed)
this_file_dir = os.path.dirname(os.path.abspath(__file__))
weights_file = os.path.join(this_file_dir, "weights/ipv3_bbox_image-54--0.12--0.96-0.18--0.95.h5")
batch_size = 32
multiview = 10
limit_upper = 1.0
limit_lower = 0.007
submission_file = "results/ipv3_bbox_image-54_multiview_%s_limit(%.3f, %.3f)_my.csv"%(multiview, limit_upper, limit_lower)
onehot_submission_file = "results/ipv3_bbox_image-54_multiview_%s_limit(%.3f, %.3f)_my_onehot.csv"%(multiview, limit_upper, limit_lower)
debug = 0

# load dataset
test_data_reader = TestDataReader()
images_test, image_names_test = test_data_reader.load_test_dataset(debug=debug)
#boxes_dict = test_data_reader.read_boxes_file()
boxes_dict = test_data_reader.read_my_boxes_file()

# create model
ipv3_bbox = Ipv3BboxImageClassifier(weights_file=weights_file)
scores_test = ipv3_bbox.predict(
    images_test,
    image_names_test,
    boxes_dict,
    batch_size=batch_size,
    verbose=1,
    nb_multiview=multiview,
    seed=seed,
)


# limit scores
scores_test = limit_upper_and_lower(scores_test, limit_upper, limit_lower)

# make submission
make_submission(submission_file, scores_test, image_names_test)

# onehot
scores_test_onehot = translate_scores_to_onehot(scores_test)
make_submission(onehot_submission_file, scores_test_onehot, image_names_test)
