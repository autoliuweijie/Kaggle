import os, sys
import numpy as np
from mylib.others import load_img_to_array
from mylib.models.bboxclassifier import Ipv3BboxClassifier, ipv3_preprocess

# settings
seed = 7
np.random.seed(seed)
batch_size = 32
input_shape = (299, 299, 3)
image_size = input_shape[:2]
samples_per_epoch = 1500000
nb_epoch = 32

file_dir = os.path.dirname(os.path.abspath(__file__))
fish_nofish_dataset_dir = os.path.join(file_dir, '../datasets/fish_dataset/')
weigts_saved_dir = os.path.join(file_dir, 'ipv3_bbox_classifier_weights/')


def create_train_images_generator(train_images_dir, image_size=None, batch_size = 32, prop=[0.5, 0.5], process_fun=None):
    '''
        Create a generator for images and labels from train_images_dir
    '''
    fish_tag = 1.0
    nofish_tag = 0.0
    
    fish_dir = os.path.join(train_images_dir, 'fish/')
    nofish_dir = os.path.join(train_images_dir, 'nofish/')
    fish_image_names = os.listdir(fish_dir)
    nofish_image_names = os.listdir(nofish_dir)
    fish_image_nb = len(fish_image_names)
    nofish_image_nb = len(nofish_image_names)
    
    fish_range = prop[0]
    nofish_range = fish_range + prop[1]
    
    while True:
        
        images, labels = [], []
        for i in range(batch_size):
            select = np.random.random()
            if select <= fish_range:
                idx = np.random.randint(fish_image_nb)
                image_name = fish_image_names[idx]
                image_url = os.path.join(fish_dir, image_name)
                labels.append(fish_tag)
            if select > fish_range and select <= nofish_range:
                idx = np.random.randint(nofish_image_nb)
                image_name = nofish_image_names[idx]
                image_url = os.path.join(nofish_dir, image_name)
                labels.append(nofish_tag)
            image = load_img_to_array(image_url, target_size=image_size)
            images.append(image)
        images = np.array(images)
        labels = np.array(labels)
        
        if process_fun is not None:
            images = process_fun(images)

        yield (images, labels)


# create train images generator
train_images_gen = create_train_images_generator(
    fish_nofish_dataset_dir, 
    image_size=image_size, 
    batch_size=batch_size,
    prop=[0.5, 0.5],
    process_fun=ipv3_preprocess,
)


# training classifier
ipv3_bbox_classifier = Ipv3BboxClassifier(input_shape=input_shape)

history = ipv3_bbox_classifier.train(
    train_images_gen,
    weights_dir_path = weigts_saved_dir,
    samples_per_epoch = samples_per_epoch,
    nb_epoch = nb_epoch,
    verbose = 1,
)

