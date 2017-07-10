'''
    Some API about load and output datas
'''
import settings
import numpy as np
import os
from keras.preprocessing import image
import csv


# settings
train_dataset_dir = settings.train_dataset_dir
test_dataset_dir = settings.test_dataset_dir
names_to_int = settings.class_names_to_int
image_size = settings.image_size


def load_train_dataset(class_names='all', img_size=image_size, train_dataset_dir=train_dataset_dir, verbose=0):
    '''
    load train dataset
    :param  class_names: a list contains the class to load, such as ['BET', 'LAG']
            verbose: no information shows if verbose=0
    :return: X_train, Y_train: np.narray (None, 3, h, w)
    '''

    if verbose != 0:
        print("Loading train dataset ...")

    # choose the class names
    if class_names == "all":
        names_list = names_to_int.keys()
    else:
        names_list = class_names

    X_train, Y_train = [], []

    for class_name in os.listdir(train_dataset_dir):

        # drouput other files
        if class_name not in names_list:
            continue

        for filename in os.listdir(os.path.join(train_dataset_dir, class_name)):
            file_url = os.path.join(os.path.join(train_dataset_dir, class_name, filename))
            img = image.load_img(file_url, target_size=img_size)
            img.convert('RGB')
            x = image.img_to_array(img, 'th')
            X_train.append(x)
            Y_train.append(names_to_int[class_name])

    X_train = np.array(X_train).astype('float32')
    Y_train = np.array(Y_train).astype('float32')

    if verbose != 0:
        print("Finish loading train dataset:")
        print "X_train: ", X_train.shape
        print "Y_train: ", Y_train.shape

    return X_train, Y_train


def load_test_dataset(test_dataset_dir=test_dataset_dir, img_size=image_size, verbose=0):
    '''
    load test dataset and filename
    :param verbose: no information shows if verbose=0
    :return: X_train: np.narray (None, 3, h, w)
             img_names:np.array (None, )
    '''

    if verbose != 0:
        print("Loading test dataset ...")

    X_test, img_names = [], []

    for img_name in os.listdir(test_dataset_dir):

        file_url = os.path.join(test_dataset_dir, img_name)
        img = image.load_img(file_url, target_size=img_size)
        x = image.img_to_array(img, 'th')
        X_test.append(x)
        img_names.append(img_name)

    X_test = np.array(X_test).astype('float32')
    img_names = np.array(img_names)

    if verbose != 0:
        print("Finish loading test dataset:")
        print "X_test: ", X_test.shape
        print "img_names: ", img_names.shape

    return X_test, img_names


def make_submission(filename, predictions, image_names):

    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
        for i in range(len(image_names)):
            line = [
                image_names[i],
                predictions[i, 0],
                predictions[i, 1],
                predictions[i, 2],
                predictions[i, 3],
                predictions[i, 4],
                predictions[i, 5],
                predictions[i, 6],
                predictions[i, 7],
            ]
            writer.writerow(line)

    print "Submission file has been generated:", filename


if __name__ == "__main__":
    # load_train_dataset(verbose=1)
    # load_test_dataset(verbose=1)
    predictions = [
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ]
    predictions = np.array(predictions)
    image_names = [
        'img001.jpg',
        'img002.jpg',
    ]
    make_submission("../caches/submission_example.csv", predictions, image_names)
