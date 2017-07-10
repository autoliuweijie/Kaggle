"""
    This script is used to split the origin dataset into train and validation parts
    @author: Liu Weijie
    @date: 2017-01-05
"""
import os
import numpy as np
import shutil


# settings
seed = 7
np.random.seed(seed)

fish_names = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

file_dir = os.path.dirname(os.path.abspath("__file__"))
url_origin_dataset = os.path.join(file_dir, 'origin_dataset/')
url_origin_dataset_train = os.path.join(url_origin_dataset, 'train/')
url_train_val_dataset = os.path.join(file_dir, 'train_val_dataset/')
url_train_dataset = os.path.join(url_train_val_dataset, 'train/')
url_val_dataset = os.path.join(url_train_val_dataset, 'val/')

split_proportion = 0.8


# mkdir
print("Create directory")
if os.path.isdir(url_train_val_dataset):
    shutil.rmtree(url_train_val_dataset)
os.mkdir(url_train_val_dataset)
os.mkdir(url_train_dataset)
os.mkdir(url_val_dataset)


# split images
print("Split Dataset")
nbr_train_samples = {}  # used for save the number of samples of different in train dataset
nbr_val_samples = {}
for fish in fish_names:

    total_images = os.listdir(os.path.join(url_origin_dataset_train, fish))
    np.random.shuffle(total_images)

    nbr_train = int(len(total_images) * split_proportion)
    train_images = total_images[:nbr_train]
    val_images = total_images[nbr_train:]

    # copy train dataset
    os.mkdir(os.path.join(url_train_dataset, fish))
    nbr_train_samples[fish] = 0
    for img in train_images:
        src = os.path.join(url_origin_dataset_train, fish, img)
        dst = os.path.join(url_train_dataset, fish, img)
        shutil.copy(src, dst)
        nbr_train_samples[fish] += 1

    # copy validation dataset
    os.mkdir(os.path.join(url_val_dataset, fish))
    nbr_val_samples[fish] = 0
    for img in val_images:
        src = os.path.join(url_origin_dataset_train, fish, img)
        dst = os.path.join(url_val_dataset, fish, img)
        shutil.copy(src, dst)
        nbr_val_samples[fish] += 1

    # copy box file in json
    if fish != 'NoF':
        src = os.path.join(url_origin_dataset_train, fish + '.json')
        dst = os.path.join(url_train_val_dataset, fish + '.json')
        shutil.copy(src, dst)

print("Finish splitting train and val images!")
print("The numbers of samples in train dataset are :")
print(nbr_train_samples)
print("The numbers of samples in val dataset are :")
print(nbr_val_samples)
