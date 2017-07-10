"""
    This script is used for training ipv3 on full images, the weights will be saved at weights/
    @author: Liu Weijie
    @date: 2017-01-07
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from models.classifiers import Ipv3FullImageClssifier
from myutils.dataio import TrainValDataReader
from keras.utils.np_utils import to_categorical


# settings
batch_size = 32
nb_epoch = 64
weights_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights/')
class_weight = {
    0: 1000.0/1375.0, 1: 1000.0/160.0, 2: 1000.0/93.0, 3: 1000.0/53.0,
    4: 1000.0/372.0, 5: 1000.0/239.0, 6: 1000.0/140.0, 7: 1000.0/587.0,
}
debug = 0


# load dataset
train_val_reader = TrainValDataReader()
images_train, labels_train, img_names = train_val_reader.load_train_dataset(verbose=1, debug=debug)
images_val, labels_val, img_names = train_val_reader.load_val_dataset(verbose=1, debug=debug)
labels_vec_train = to_categorical(labels_train)
labels_vec_val = to_categorical(labels_val)

# create and train ipv3
ipv3_full = Ipv3FullImageClssifier()
ipv3_full.train(
    images_train,
    labels_vec_train,
    weights_file=weights_file,
    class_weight=class_weight,
    validation_data=(images_val, labels_vec_val),
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    verbose=1
)

