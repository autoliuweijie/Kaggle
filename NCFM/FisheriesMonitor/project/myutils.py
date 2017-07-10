import numpy as np
import settings
from data_managers import load_train_dataset, load_test_dataset, make_submission
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Flatten, AveragePooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import selectivesearch as sls
from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import csv
import pickle


def serialize_array(filename, np_array):
    json_text = pickle.dumps(np_array)
    with open(filename, 'wb') as f:
        f.write(json_text)
    return None

def load_array(filename):
    with open(filename, 'rb') as f:
        json_text = f.read()
    array = pickle.loads(json_text)
    return array
    

# a util for read results.csv
def read_csv_result(filename):
    scores = []
    img_names = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            
            if line[0][0:3] != 'img':
                continue
            
            scores.append([float(p) for p in line[1:]])
            img_names.append(line[0])
            
    return np.array(scores), np.array(img_names)


def sort_by_value(dictory): 
    items=dictory.items() 
    backitems=[[v[1], v[0]] for v in items] 
    backitems.sort() 
    return [backitems[i][1] for i in range(0,len(backitems))]


def image_resize(image, size):
    img = np.rollaxis(image, 0, 3)
    img = cv.resize(img, size)
    img = np.rollaxis(img, 2, 0)
    return img
        


def extract_regions(image, scale=500, min_pixels=1000):
    '''
    img: np.narray in tf format
    return: list [[x, y, w, h], ...,]
    '''
    img = image.copy()
    img = img.astype('uint8')  # for fast
    img = np.rollaxis(img, 0, 3)  # change to dim_ordering to tf
    
    # get roi
    img_lbl, region = sls.selective_search(img, scale=scale, sigma=0.8, min_size=50)
    
    candidates = set()
    for r in region:  
        # excluding same rectangle (with different segments)
        if r['rect'] in set():
            continue        
        # excluding regions smaller than 2000 pixels
        if r['size'] < min_pixels:
            continue       
        candidates.add(r['rect'])
    
    return list(candidates)


def preprocessing_for_ipv3(X):
    X_normal = (X / 255.0 - 0.5) * 2
    return X_normal


def average_predict_in_augument(model, X_test, nb_average=5, verbose=0, random_seed=7):
    
    # create image generator
    test_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    test_datagen.fit(X_test)
    
    # predict for each image
    num_data = len(X_test)
    Y_test_ave = []
    for i in range(num_data):
        
        if verbose != 0 and i%200 == 0:
            print("%s / %s"%(i, num_data))
        
        # buid a data generator iterator for this image
        test_generator = test_datagen.flow(
            X_test[i:i+1],
            np.zeros((1, )),
            batch_size=nb_average,
            seed=random_seed,
            shuffle=False,  # important!
        )
        
        Y_pre = model.predict_generator(test_generator, nb_average) # predict in different noise       
        Y_pre = np.average(Y_pre, axis=0) # average predictions    
        Y_test_ave.append(Y_pre)
    
    return np.array(Y_test_ave)


def create_inception_v3(input_size=(299, 299)):
    
    InceptionV3_notop = InceptionV3(
        include_top=False, 
        weights='imagenet',
        input_tensor=None, 
        input_shape=(3, input_size[0], input_size[1])
    )
    
    output = InceptionV3_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)
    output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dense(8, activation='softmax', name='predictions')(output)
    InceptionV3_model = Model(InceptionV3_notop.input, output)
    optimizer = SGD(lr = 0.0001, momentum = 0.9, decay = 0.0, nesterov = True)
    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    
    return InceptionV3_model