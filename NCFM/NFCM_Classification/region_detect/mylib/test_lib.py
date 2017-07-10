'''
	This script is used for test lib modules
'''
import sys, os
from dataio import TrainDataReader, TestDataReader
from models.bboxregressor import Ipv3BboxRegressor

file_dir_path = os.path.dirname(os.path.abspath(__file__))
nb_bboxes = 5
weigts_dir = os.path.join(file_dir_path, 'test_dir')
debug = 1


train_data_reader = TrainDataReader()
images_train, image_names_train, bboxs_train = train_data_reader.load_train_image_bbox_imgname(debug=debug)
Test_data_reader = TestDataReader()
images_test, image_names_test, bboxs_test = Test_data_reader.load_test_image_bbox_imgname(debug=debug)


ipv3_bbox = Ipv3BboxRegressor(nb_bbox=nb_bboxes)
history = ipv3_bbox.train(images_train, bboxs_train, weights_dir_path=weigts_dir)

