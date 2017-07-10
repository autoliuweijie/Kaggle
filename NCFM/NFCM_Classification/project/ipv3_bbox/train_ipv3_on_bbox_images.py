"""
    This script is used for training ipv3 on bbox images, the weights will be saved at weights/
    @author: Liu Weijie
    @date: 2017-01-09
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from models.classifiers import Ipv3BboxImageClassifier
from myutils.dataio import TrainValDataReader
from keras.utils.np_utils import to_categorical
import numpy as np


# config
file_dir = os.path.dirname(os.path.abspath(__file__))
seed = 7
np.random.seed(seed)
debug = 0


# settings
batch_size = 32
nb_epoch = 64
weights_file = os.path.join(file_dir, 'weights/')
class_weight = {
    0: 1000.0/1375.0, 1: 1000.0/160.0, 2: 1000.0/93.0, 3: 1000.0/53.0,
    4: 1000.0/372.0, 5: 1000.0/239.0, 6: 1000.0/140.0, 7: 1000.0/587.0,
}


# load train and val dataset and bbox file
train_val_reader = TrainValDataReader()
images_train, labels_train, img_names_train = train_val_reader.load_train_dataset(verbose=1, debug=debug)
images_val, labels_val, img_names_val = train_val_reader.load_val_dataset(verbose=1, debug=debug)
labels_vec_train = to_categorical(labels_train)
labels_vec_val = to_categorical(labels_val)
boxes_trin_val = train_val_reader.read_boxes_file()


# create and train ipv3
ipv3_bbox = Ipv3BboxImageClassifier()
ipv3_bbox.train(
    images_train,
    labels_train,
    img_names_train,
    boxes_trin_val,
    validation_data=(images_val, labels_val, img_names_val),
    weights_file=weights_file,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    class_weight=class_weight,
    verbose=1,
)


