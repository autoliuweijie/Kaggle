'''
    This is a module contains different kinds of classifier
'''
import numpy as np
import os
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras import backend
import cv2 as cv
from keras.utils.np_utils import to_categorical
from others import unison_shuffled_copies
from myutils import settings
from keras.preprocessing.image import array_to_img, img_to_array
import matplotlib.pyplot as plt


# settings
backend.set_image_dim_ordering('tf')
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 32
NB_EPOCH = 16
NB_MULTIVIEW = 10
RANDOM_SEED = 7
NAME_TO_INT = settings.fishname_to_int


class AbstractClassifier(object):
    '''
    Classifier based on keras
    '''

    def __init__(self, weights=None):
        self.model = None
        if weights is not None and self.model is not None:
            self.model.load_weights(weights)

    def train(self, images, labels_vec, image_names, bbox_dict, weights_file, batch_size=BATCH_SIZE, validation_data=None, class_weight=None, verbose=0):
        pass

    def evaluate(self, X, Y):
        pass

    def predict(self, images, images_name, bbox_dict, batch_size=BATCH_SIZE, verbose=0, nb_multiview=NB_MULTIVIEW, seed=RANDOM_SEED):
        pass


class Ipv3FullImageClssifier(AbstractClassifier):
    '''
    This is a classifier on full image based on GoogLeNet of Inception v3.
    '''
    ipv3_input_shape = (299, 299, 3)
    weights_saved_dir = os.path.join(FILE_DIR, 'weights/ipv3_full/')
    datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    def __init__(self, weights_file=None):

        # create inception v3 model
        ipv3_notop = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=self.ipv3_input_shape
        )
        output = ipv3_notop.get_layer(index=-1).output
        output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
        output = Flatten(name='flatten')(output)
        output = Dense(8, activation='softmax', name='predictions')(output)
        self.model = Model(ipv3_notop.input, output)
        optimizer = SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])

        if weights_file is not None: self.model.load_weights(weights_file)

    def train(self, images, labels_vec, image_names=None, bbox_dict=None, weights_file=weights_saved_dir, batch_size=BATCH_SIZE, validation_data=None,
              class_weight=None, nb_epoch=NB_EPOCH, verbose=0):
        if verbose: print("Start Training.")

        # resize, balance and preprocess images
        images = self.__resize_images_for_ipv3(images)
        images = self.__preprocessing_for_ipv3(images)

        # Data augumentation
        train_generator = self.datagen.flow(images, labels_vec, batch_size=batch_size)

        images_val, labels_vec_val = validation_data
        images_val = self.__resize_images_for_ipv3(images_val)
        images_val = self.__preprocessing_for_ipv3(images_val)
        weights_saved_file = os.path.join(weights_file,
                                          'ipv3_full_image-{epoch:02d}--{val_loss:.2f}--{val_acc:.2f}.h5')
        checkpoint = ModelCheckpoint(weights_saved_file, monitor='val_loss', verbose=verbose, save_best_only=False)

        history = self.model.fit_generator(
            train_generator,
            samples_per_epoch=len(images),
            validation_data=(images_val, labels_vec_val),
            nb_epoch=nb_epoch,
            callbacks=[checkpoint],
            verbose=verbose,
            class_weight=class_weight
        )

        if verbose: print("Finsh Training!")

        return history

    def predict(self, images, images_name=None, bbox_dict=None, batch_size=BATCH_SIZE, verbose=0, nb_multiview=NB_MULTIVIEW, seed=RANDOM_SEED):
        # resize, balance and preprocess images
        images = self.__resize_images_for_ipv3(images)
        images = self.__preprocessing_for_ipv3(images)

        nbr_images = len(images)
        y_multiview = []
        for i in range(nbr_images):
            if verbose != 0 and i % 200 == 0:  # print per 200 images
                print("%s / %s"%(i, nbr_images))

            test_generator = self.datagen.flow(
                images[i: i+1],
                np.zeros((1, )),
                batch_size=nb_multiview,
                seed=seed,
                shuffle=False,
            )

            y = self.model.predict_generator(test_generator, nb_multiview)
            y = np.average(y, axis=0)
            y_multiview.append(y)

        return np.array(y_multiview)


    def __preprocessing_for_ipv3(self, X):
        X_normal = (X / 255.0 - 0.5) * 2
        return X_normal

    def __resize_images_for_ipv3(self, images):
        resize_images = []
        for i in range(len(images)):
            image = images[i]
            image = array_to_img(image)
            image = image.resize(self.ipv3_input_shape[:2])
            image = img_to_array(image)
            resize_images.append(image)
        return np.array(resize_images)


class Ipv3BboxImageClassifier(AbstractClassifier):
    '''
    This is a classifier on the bounding box of image with GoogLNet constructed by Inception V3
    '''
    ipv3_input_shape = (299, 299, 3)
    weights_saved_dir = os.path.join(FILE_DIR, 'weights/ipv3_bbox/')
    datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    nb_classes = 8   # 8 is the number of fish classes

    def __init__(self, weights_file=None):

        # create inception v3 model
        ipv3_notop = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=self.ipv3_input_shape
        )
        output = ipv3_notop.get_layer(index=-1).output
        output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
        output = Flatten(name='flatten')(output)
        output = Dense(8, activation='softmax', name='predictions')(output)
        self.model = Model(ipv3_notop.input, output)
        optimizer = SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])

        if weights_file is not None: self.model.load_weights(weights_file)

    def train(self, images, labels_vec, image_names, bbox_dict, validation_data, weights_file=weights_saved_dir,
              batch_size=BATCH_SIZE, class_weight=None, nb_epoch=NB_EPOCH, verbose=0):

        if verbose: print("Start Training!")

        # get bbox train detaset
        images_bbox_train, labels_bbox_train = self.__extract_bbox(images, image_names, bbox_dict)
        images_bbox_val, labels_bbox_val = self.__extract_bbox(validation_data[0], validation_data[2], bbox_dict)

        if verbose:
            print("Train bbox: %s"%(images_bbox_train.shape))
            print("Val bbox: %s"%(images_bbox_val.shape))

        # preprocess bbox images
        images_bbox_train = self.__resize_images_for_ipv3(images_bbox_train)
        images_bbox_train = self.__preprocessing_for_ipv3(images_bbox_train)
        images_bbox_val = self.__resize_images_for_ipv3(images_bbox_val)
        images_bbox_val = self.__preprocessing_for_ipv3(images_bbox_val)

        # Data augumentation
        train_generator = self.datagen.flow(images_bbox_train, labels_bbox_train, batch_size=batch_size)

        # train model
        weights_saved_file = os.path.join(
            weights_file,
            'ipv3_bbox_image-{epoch:02d}--{loss:.2f}--{acc:.2f}-{val_loss:.2f}--{val_acc:.2f}.h5'
        )
        checkpoint = ModelCheckpoint(weights_saved_file, monitor='val_loss', verbose=verbose, save_best_only=False)

        history = self.model.fit_generator(
            train_generator,
            samples_per_epoch=len(images),
            validation_data=(images_bbox_val, labels_bbox_val),
            nb_epoch=nb_epoch,
            callbacks=[checkpoint],
            verbose=verbose,
            class_weight=class_weight
        )

        if verbose: print("Finsh Training!")

        return history

    def predict(self, images, images_name, bbox_dict, batch_size=BATCH_SIZE, verbose=0, nb_multiview=NB_MULTIVIEW, seed=RANDOM_SEED):

        nb_images = len(images)
        y_multiview = []
        for i in range(nb_images):

            if verbose != 0 and i % 200 == 0:  # print per 200 images
                print("%s / %s"%(i, nb_images))

            # get image
            image = images[i]
            image_name = images_name[i]
            boxes = bbox_dict[image_name]

            # for each bbox
            y_list = []
            for box_info in boxes:

                # get image_box
                x, y, w, h, c = box_info['x'], box_info['y'], box_info['width'], box_info['height'], box_info['class']
                if x < 0 or y<0 or w<=0 or h<=0:
                    continue
                else:
                    image_box = image[y: y+h, x: x+w]

                # preprocess bbox
                image_box = np.array([image_box])
                image_box = self.__resize_images_for_ipv3(image_box)
                image_box = self.__preprocessing_for_ipv3(image_box)

                # multiview
                test_generator = self.datagen.flow(
                    image_box,
                    np.zeros((1,)),
                    batch_size=nb_multiview,
                    seed=seed,
                    shuffle=False,
                )

                # predict
                y = self.model.predict_generator(test_generator, nb_multiview)
                y_list.append(np.average(y, axis=0))

            # average result from each bbox
            if len(y_list) > 0:
                y_list = np.array(y_list)
                y_multiview.append(np.average(y_list, axis=0))
            else:
                y_multiview.append(self.__fishname_to_vec('NoF'))

        return np.array(y_multiview)

    def __extract_bbox(self, images, image_names, bbox_dict):

        nb_images = len(images)
        images_bbox = []
        labels_bbox = []
        for i in range(nb_images):

            image = images[i]

            try:
                boxes_list = bbox_dict[image_names[i]]
            except KeyError:
                continue

            for box_cor in boxes_list:
                x, y, w, h, fish = box_cor['x'], box_cor['y'], box_cor['width'], box_cor['height'], box_cor['class']
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    print("Wrong bbox in %s"%(image_names[i]))
                    continue
                images_bbox.append(image[y: y+h, x: x+w])
                labels_bbox.append(self.__fishname_to_vec(fish))

        images_bbox = np.array(images_bbox)
        labels_bbox = np.array(labels_bbox)

        return images_bbox, labels_bbox

    def __fishname_to_vec(self, fishname):
        vec = to_categorical([NAME_TO_INT[fishname]], nb_classes=self.nb_classes)[0]
        return vec

    def __preprocessing_for_ipv3(self, X):
        X_normal = (X / 255.0 - 0.5) * 2
        return X_normal

    def __resize_images_for_ipv3(self, images):
        resize_images = []
        for i in range(len(images)):
            image = images[i]
            image = array_to_img(image)
            image = image.resize(self.ipv3_input_shape[:2])
            image = img_to_array(image)
            resize_images.append(image)
        return np.array(resize_images)


if __name__ == "__main__":
    ipve_full = Ipv3FullImageClssifier()
