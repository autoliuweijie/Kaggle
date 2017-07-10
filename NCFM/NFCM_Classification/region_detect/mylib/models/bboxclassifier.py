"""
    This module is used for bbox classification of fish or nofish
    @author: Liu Weijie
    @date: 2017-01-27
"""
import sys, os
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D, Reshape
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras import backend
from mylib.others import resize_image


backend.set_image_dim_ordering('tf')


BATCH_SIZE = 32
NB_EPOCH = 16
NB_BBOX = 5


class AbstractBboxClassifier(object):

    def __init__(self, weights_file, input_shape, verbose):
        pass

    def train(self, train_images_generator, weights_dir_path, nb_epoch, samples_per_epoch, verbose):
        pass

    def predict(self, images, batch_size):
        pass


class Ipv3BboxClassifier(object):

    ipv3_input_shape = (299, 299, 3)

    def __init__(self, weights_file=None, input_shape=ipv3_input_shape, verbose=0):
        
        self.input_shape = input_shape

        # create inception v3 model
        ipv3_notop = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=input_shape
        )
        output = ipv3_notop.get_layer(index=-1).output
        output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
        output = Flatten(name='flatten')(output)
        output = Dense(512, activation='relu', name='dense0')(output)
        output = Dense(128, init='normal', activation='relu', name='dense1')(output)
        output = Dense(32, init='normal', activation='relu', name='dense2')(output)
        output = Dense(1, init='normal', activation='sigmoid', name='dense3')(output)
        self.model = Model(ipv3_notop.input, output)
        optimizer = SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        if verbose: print self.model.summary()

        if weights_file is not None: self.model.load_weights(weights_file)

    def train(self, train_images_generator, weights_dir_path, samples_per_epoch, nb_epoch=NB_EPOCH, verbose=0):

        if verbose: print "Start Training!"

        weights_saved_file = os.path.join(
            weights_dir_path,
            'ipv3_bbox_classifier-{epoch:02d}--{loss:.2f}--{acc:.2f}.h5',
        )

        checkpoint = ModelCheckpoint(weights_saved_file, monitor='accuracy', verbose=verbose, save_best_only=False)
      
        history = self.model.fit_generator(
            train_images_generator,
            samples_per_epoch = samples_per_epoch,
            nb_epoch = nb_epoch,
            callbacks = [checkpoint]
        )

        if verbose: print "Finish Training!"
        return history
    
    def predict(self, images, batch_size):
        
        resize_images = []
        for i in range(len(images)):
            image = images[i]
            image = resize_image(image, self.input_shape)
            resize_images.append(image)
        images = np.array(resize_images)
        images = ipv3_preprocess(images)
        results = self.model.predict(images, batch_size=batch_size).reshape((len(images),))
        return results
        


def ipv3_preprocess(images):
    images_normal = (images / 255.0 - 0.5) * 2
    return images_normal














