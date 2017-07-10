"""
    This is used for test ../models.classifiers.Ipv3BboxImageClssifier
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from models.classifiers import Ipv3BboxImageClassifier
from myutils.dataio import TrainValDataReader
from keras.utils.np_utils import to_categorical


train_val_reader = TrainValDataReader()
images_train, labels_train, img_names_train = train_val_reader.load_train_dataset(verbose=1, debug=1)
images_val, labels_val, img_names_val = train_val_reader.load_val_dataset(verbose=1, debug=1)
labels_vec_train = to_categorical(labels_train)
labels_vec_val = to_categorical(labels_val)
boxes_trin_val = train_val_reader.read_boxes_file()


ipv3_bbox = Ipv3BboxImageClassifier()
ipv3_bbox.train(images_train,
                labels_train,
                img_names_train,
                boxes_trin_val,
                validation_data=(images_val, labels_val, img_names_val),
                verbose=1,
                )
