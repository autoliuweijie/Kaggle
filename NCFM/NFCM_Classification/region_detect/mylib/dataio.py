"""
	This module is used for data input or output
	@author: Liu Weijie
	@date: 2017-01-11
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from others import load_img_to_array
import settings


FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ORIGIN_DATASET_DIR_PATH = os.path.join(FILE_DIR_PATH, '../../datasets/origin_dataset/')
FISHNAME_TO_INT = settings.fishname_to_int
HAS_FISH_TAG = 100.0
NO_FISH_TAG = 0.0


class TrainDataReader(object):

    default_dataset_dir_path = ORIGIN_DATASET_DIR_PATH
    default_nb_bbox = 5


    def __init__(self, dataset_dir_path=default_dataset_dir_path):
        self.dataset_dir_path = dataset_dir_path

    def load_train_image_bbox_imgname(self, nb_bbox=default_nb_bbox, debug=0):
        print("Start loading train data!")

        dataset_train_dir_path = os.path.join(self.dataset_dir_path, 'train/')

        # load images and their name
        images, image_names = [], []
        for fishname in os.listdir(dataset_train_dir_path):

            if fishname not in FISHNAME_TO_INT.keys():
                continue

            fish_dir_path = os.path.join(dataset_train_dir_path, fishname)
            for imgname in os.listdir(fish_dir_path):
                image_names.append(imgname)
                image_path = os.path.join(fish_dir_path, imgname)
                images.append(load_img_to_array(image_path,  dim_ordering='tf'))

                if debug: break  # read only one image
        images = np.array(images)

        # load bbox file
        boxes = {}
        for filename in os.listdir(dataset_train_dir_path):

            if filename[-5:] != '.json':
                continue

            file_path = os.path.join(dataset_train_dir_path, filename)
            with open(file_path, 'r') as json_file:
                json_str = json_file.read()
            annot_list = None
            exec "annote_list = " + json_str

            # read each images
            for annote in annote_list:
                filename = annote['filename'][-13:]
                boxes[filename] = annote['annotations']

        # trans boxes to list consistent with image_names
        bboxs = []
        for imgname in image_names:
            annote = boxes[imgname]
            bbox = []
            for i in range(nb_bbox):

                try:
                    box_info = annote[i]
                    x, y, w, h = box_info['x'], box_info['y'], box_info['width'], box_info['height']
                    bbox.append([x, y, w, h, HAS_FISH_TAG])
                except IndexError: # NoF
                    bbox.append([0, 0, 0, 0, NO_FISH_TAG])

            bboxs.append(bbox)
        bboxs = np.array(bboxs)
        image_names = np.array(image_names)

        print("Finish loading train data!")
        
        return images, image_names, bboxs


class TestDataReader(object):

    default_dataset_dir_path = ORIGIN_DATASET_DIR_PATH
    test_img_dir = 'test_stg1/'
    bbox_filename = 'test_stg1.json'
    default_nb_bbox = 5

    def __init__(self, dataset_dir_path=default_dataset_dir_path):
        self.dataset_dir_path = dataset_dir_path
        self.dataset_test_img_dir_path = os.path.join(self.dataset_dir_path, self.test_img_dir)
        self.bbox_filename_path = os.path.join(self.dataset_dir_path, self.bbox_filename)

    def load_test_image_bbox_imgname(self, nb_bbox=default_nb_bbox, debug=0):
        print("Start loading test dataset!")

        # load image and their names
        images, image_names = [], []
        for filename in os.listdir(self.dataset_test_img_dir_path):
            image_names.append(filename)
            image_path = os.path.join(self.dataset_test_img_dir_path, filename)
            images.append(load_img_to_array(image_path,  dim_ordering='tf'))
            if debug and len(image_names)==10: break  # debug model only read 10 images
        images = np.array(images)
        
        # load bbox file
        boxes = {}
        with open(self.bbox_filename_path, 'r') as json_file:
            json_str = json_file.read()
        annot_list = None
        exec "annote_list = " + json_str

        # read each images
        for annote in annote_list:
            filename = annote['filename'][-13:]
            boxes[filename] = annote['annotations']

        # trans boxes to list consistent with image_names
        bboxs = []
        for imgname in image_names:
            annote = boxes[imgname]
            bbox = []
            for i in range(nb_bbox):

                try:
                    box_info = annote[i]
                    x, y, w, h = box_info['x'], box_info['y'], box_info['width'], box_info['height']
                    bbox.append([x, y, w, h, HAS_FISH_TAG])
                except IndexError: # NoF
                    bbox.append([0, 0, 0, 0, NO_FISH_TAG])

            bboxs.append(bbox)

        bboxs = np.array(bboxs)
        image_names = np.array(image_names)

        print("Finish loading test data!")
        
        return images, image_names, bboxs

