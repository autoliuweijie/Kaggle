"""
    This script is used for training ipv3 bbox regressor
"""

import sys, os
from mylib.dataio import TrainDataReader, TestDataReader
from mylib.models.bboxregressor import Ipv3BboxRegressor


file_dir_path = os.path.dirname(os.path.abspath(__file__))
nb_bboxes = 5
weigts_dir = os.path.join(file_dir_path, 'ipv3_bbox_regressor_weights_2/')
init_weights_file = os.path.join(file_dir_path, 'ipv3_bbox_regressor_weights/ipv3_bbox_regressor_nb_5-255--18.76-22.52.h5')
batch_size = 32
nb_epoch = 256
debug = 0


train_data_reader = TrainDataReader()
images_train, image_names_train, bboxs_train = train_data_reader.load_train_image_bbox_imgname(debug=debug)
Test_data_reader = TestDataReader()
images_test, image_names_test, bboxs_test = Test_data_reader.load_test_image_bbox_imgname(debug=debug)


ipv3_bbox = Ipv3BboxRegressor(weights_file=init_weights_file, nb_bbox=nb_bboxes)
history = ipv3_bbox.train(
    images_train,
    bboxs_train,
    weights_dir_path=weigts_dir,
    validation_set=(images_test, bboxs_test),
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    verbose=1,
    )
