"""
    This is used for test ../models.classifiers.Ipv3FullImageClssifier
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath("__file__")), '../'))
from models.classifiers import Ipv3FullImageClssifier
from myutils.dataio import TrainValDataReader
from keras.utils.np_utils import to_categorical


train_val_reader = TrainValDataReader()
images_train, labels_train, img_names = train_val_reader.load_train_dataset(verbose=1, debug=1)
images_val, labels_val, img_names = train_val_reader.load_val_dataset(verbose=1, debug=1)
labels_vec_train = to_categorical(labels_train)
labels_vec_val = to_categorical(labels_val)

ipv3_full = Ipv3FullImageClssifier()
ipv3_full.train(images_train, labels_vec_train, validation_data=(images_val, labels_vec_val), batch_size=32, verbose=1)
