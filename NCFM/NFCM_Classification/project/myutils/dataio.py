'''
    This is a module about data I/O
    @author: Liu Weijie
    @date: 20150105
'''
import numpy as np
import os
import settings
import cv2 as cv
import csv


# settings
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
URL_TRAIN_VAL_DATASET = os.path.join(FILE_DIR, '../../datasets/train_val_dataset/')
URL_ORIGIN_DATASET = os.path.join(FILE_DIR, '../../datasets/origin_dataset')
NAME_TO_INT = settings.fishname_to_int


def read_img(url):
    '''
    use opencv to read images, and return a [height, width, channal(rgb)] format image
    '''
    img = cv.imread(url)[:, :, ::-1].astype('float32')
    return img


class TrainValDataReader(object):
    '''
    A class used for read dataset
    '''

    def __init__(self, url_train_val_dataset=URL_TRAIN_VAL_DATASET):
        self.url_dataset = url_train_val_dataset
        self.url_train_dataset = os.path.join(self.url_dataset, 'train/')
        self.url_val_dataset = os.path.join(self.url_dataset, 'val/')

    def load_train_dataset(self, verbose=0, debug=0):
        if verbose: print("Start loading train dataset...")
        images, labels, img_names = self.__load_dataset(self.url_train_dataset, debug=debug)
        if verbose:print("Finish loading train dataset!")
        return images, labels, img_names

    def load_val_dataset(self, verbose=0, debug=0):
        if verbose: print("Start loading val dataset...")
        images, labels, img_names = self.__load_dataset(self.url_val_dataset, debug=debug)
        if verbose:print("Finish loading val dataset!")
        return images, labels, img_names

    def read_boxes_file(self):

        boxes = {}
        for file in os.listdir(self.url_dataset):

            # read json file
            if file[-5:] != '.json': continue
            url_json = os.path.join(self.url_dataset, file)
            with open(url_json, 'r') as file_json:
                json_text = file_json.read()
            annote_list = None
            exec "annote_list = " + json_text

            # read each images
            for annote in annote_list:
                filename = annote['filename'][-13:]
                boxes[filename] = annote['annotations']

        return boxes

    def __load_dataset(self, url_dataset, debug=0):
        images, labels, img_names = [], [], []
        for fish in NAME_TO_INT.keys():
            path_images = os.path.join(url_dataset, fish)
            for image_name in os.listdir(path_images):
                url_img = os.path.join(path_images, image_name)
                images.append(read_img(url_img))
                labels.append(NAME_TO_INT[fish])
                img_names.append(image_name)
                if debug: break  # only load one image each class of fish
        return np.array(images), np.array(labels), np.array(img_names)


class TestDataReader(object):

    url_origin_dataset_test = os.path.join(URL_ORIGIN_DATASET, 'test_stg1/')
    url_test_json_file = os.path.join(URL_ORIGIN_DATASET, 'test_stg1.json')
    url_my_test_json_file = os.path.join(URL_ORIGIN_DATASET, 'my_test_stg1.json')

    def load_test_dataset(self, debug=0):
        images, img_names = [], []
        nb_images = 0
        for image_name in os.listdir(self.url_origin_dataset_test):
            url_img = os.path.join(self.url_origin_dataset_test, image_name)
            images.append(read_img(url_img))
            img_names.append(image_name)

            nb_images += 1
            if debug and nb_images == 10: break  # only load one image if debug
        return np.array(images), np.array(img_names)

    def read_boxes_file(self):

        boxes = {}
        with open(self.url_test_json_file, 'r') as json_file:
            json_text = json_file.read()
            annote_list = None
            exec "annote_list = " + json_text

        for annote in annote_list:
            filename = annote['filename'][-13:]
            boxes[filename] = annote['annotations']

        return boxes
    
    def read_my_boxes_file(self):
        import json
        
        boxes = {}
        with open(self.url_my_test_json_file, 'r') as json_file:
            str_json = json_file.read()
            json_dict = json.loads(str_json)
        
        for filename in json_dict.keys():
            bboxes = json_dict[filename]
            annotations = []
            for bbox in bboxes:
                x, y, w, h, s = bbox
                annotations.append({
                        'class': "UNKNOW",
                        'height': h,
                        'width': w,
                        'x': x,
                        'y': y                
                    })
            boxes[filename] = annotations
        return boxes
        

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


def limit_upper_and_lower(scores, upper, lower):
    scores[scores>=upper] = upper
    scores[scores<=lower] = lower
    return scores

def translate_scores_to_onehot(scores):
    onehot = np.zeros_like(scores)
    onehot[range(len(scores)), np.argmax(scores, axis=1)] = 1.0
    return onehot


if __name__ == "__main__":
    # train_val_reader = TrainValDataReader()
    # images, labels, img_names = train_val_reader.load_train_dataset(verbose=1, debug=1)
    # print images.shape, images[1].shape, labels.shape, img_names.shape
    # images, labels, img_names = train_val_reader.load_val_dataset(verbose=1, debug=1)
    # print images.shape, images[1].shape, labels.shape, img_names.shape
    # print train_val_reader.read_boxes_file()
    scores = np.array([[0.7, 0.2, 0.1], [0.4, 0.2, 0.5]])
    print translate_scores_to_onehot(scores)
