"""
    This module is used for bbox regression
    @author: Liu Weijie
    @date: 2017-01-12
"""
import sys, os
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D, Reshape
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras import backend


backend.set_image_dim_ordering('tf')


BATCH_SIZE = 32
NB_EPOCH = 16
NB_BBOX = 5


class AbstractBboxRegressor(object):

    default_batch_size = BATCH_SIZE
    default_nb_epoch = NB_EPOCH
    default_nb_bbox = NB_BBOX

    def __init__(self, weights_file, nb_bbox):
        pass

    def train(self, images_train, bboxes_train, weights_dir_path, validation_set=None, batch_size=default_nb_epoch, nb_epoch=default_nb_epoch, verbose=0):
        pass

    def predict(self, images, batch_size=default_nb_epoch):
        pass

    def evaluate(self, images, bboxes):
        pass


class Ipv3BboxRegressor(AbstractBboxRegressor):

    ipv3_input_shape = (299, 299, 3)

    def __init__(self, weights_file=None, nb_bbox=AbstractBboxRegressor.default_nb_bbox):

        self.nb_bbox = nb_bbox

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
        output = Dense(5*nb_bbox, activation='relu', name='predictions')(output)
        output = Reshape((nb_bbox, 5), name='reshape')(output)
        self.model = Model(ipv3_notop.input, output)
        optimizer = SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=True)
        self.model.compile(loss='mean_absolute_error', optimizer = optimizer, metrics=['mae', 'mse'])

        if weights_file is not None: self.model.load_weights(weights_file)

    def train(self, images_train, bboxes_train, weights_dir_path, validation_set=None, batch_size=AbstractBboxRegressor.default_batch_size, nb_epoch=AbstractBboxRegressor.default_nb_epoch, verbose=1):
        if verbose: print("Start Training.")

        # resize, balance and preprocess images
        images_train = self.__resize_images_for_ipv3(images_train)
        images_train = self.__preprocessing_for_ipv3(images_train)


        # training model
        if validation_set is None:

            if verbose: print "    Training without validation"
            weights_saved_file = os.path.join(
                weights_dir_path,
                'ipv3_bbox_regressor_nb_%s-{epoch:02d}--{loss:.2f}.h5'%(self.nb_bbox),
            )
            checkpoint = ModelCheckpoint(weights_saved_file, monitor='loss', verbose=verbose, save_best_only=False)

            history = self.model.fit(
                images_train, 
                bboxes_train, 
                batch_size=batch_size, 
                nb_epoch=nb_epoch, 
                callbacks=[checkpoint],
                verbose=verbose
                )


        else:

            if verbose: print "    Training with validation"

            # get val dataset
            images_val, bboxes_val = validation_set
            images_val = self.__resize_images_for_ipv3(images_val)
            images_val = self.__preprocessing_for_ipv3(images_val)

            weights_saved_file = os.path.join(
                weights_dir_path,
                'ipv3_bbox_regressor_nb_%s-{epoch:02d}--{loss:.2f}-{val_loss:.2f}.h5'%(self.nb_bbox),
            )
            checkpoint = ModelCheckpoint(weights_saved_file, monitor='val_loss', verbose=verbose, save_best_only=False)

            history = self.model.fit(
                images_train, 
                bboxes_train, 
                validation_data=(images_val, bboxes_val), 
                batch_size=batch_size, 
                nb_epoch=nb_epoch, 
                callbacks=[checkpoint],
                verbose=verbose
                )

        if verbose: print("Finsh Training!")

        return history

    def predict(self, images, batch_size=AbstractBboxRegressor.default_nb_epoch):

        # resize, balance and preprocess images
        images = self.__resize_images_for_ipv3(images)
        images = self.__preprocessing_for_ipv3(images)

        # predict
        boxes = self.model.predict(images, batch_size=batch_size)
               
        return boxes
    
    def predict_with_slide_bbox(self, images, batch_size=AbstractBboxRegressor.default_nb_epoch, slide_rate=0.33, min_size=10, min_p=50):
    
        all_bboxes = self.predict(images, batch_size=batch_size)
        
        new_all_bboxes = []
        for i in range(len(images)):
            image = images[i]
            bboxes = all_bboxes[i]
            bboxes = slide_bboxes(bboxes, image, slide_rate=slide_rate, min_size=min_size, min_p=min_p)
            new_all_bboxes.append(bboxes)
        
        return new_all_bboxes

    def __preprocessing_for_ipv3(self, X):
        X_normal = (X / 255.0 - 0.5) * 2
        return X_normal

    def __resize_images_for_ipv3(self, images):
        from others import img_to_array, array_to_img
        resize_images = []
        for i in range(len(images)):
            image = images[i]
            image = array_to_img(image)
            image = image.resize(self.ipv3_input_shape[:2])
            image = img_to_array(image)
            resize_images.append(image)
        return np.array(resize_images)


def slide_bboxes(bboxes, image, slide_rate=0.33, min_size=10, min_p = 50):
    h_img, w_img = image.shape[:2]
    new_bboxes = []
    
    for bbox in bboxes:
        
        x, y, w, h, p = bbox[:]

        if (x <=10 and y <=10) or (h < 1 or w < 1) or (h*w < min_size) or (p < min_p):
            continue

        delta_x = w * slide_rate
        delta_y = h * slide_rate

        x_nb_pos = int((w_img - x) / delta_x)
        y_nb_pos = int((h_img - y) / delta_y)
        x_nb_neg = int(x / delta_x) + 1
        y_nb_neg = int(y / delta_y) + 1

        for x_idx in range(x_nb_pos):
            for y_idx in range(y_nb_pos):
                new_bboxes.append([
                        x + x_idx*delta_x,
                        y + y_idx*delta_y,
                        w,
                        h,
                        p,
                    ])
        for x_idx in range(x_nb_neg):
            for y_idx in range(y_nb_neg):
                new_bboxes.append([
                        x - x_idx*delta_x,
                        y - y_idx*delta_y,
                        w,
                        h,
                        p,
                    ])
        for x_idx in range(x_nb_neg):
            for y_idx in range(y_nb_pos):
                new_bboxes.append([
                        x - x_idx*delta_x,
                        y + y_idx*delta_y,
                        w,
                        h,
                        p,
                    ])
        for x_idx in range(x_nb_pos):
            for y_idx in range(y_nb_neg):
                new_bboxes.append([
                        x + x_idx*delta_x,
                        y - y_idx*delta_y,
                        w,
                        h,
                        p,
                    ])
    
    return new_bboxes