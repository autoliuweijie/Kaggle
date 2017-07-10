'''
    Some settings
'''

# about data I/O
dataset_dir = "../dataset/"
train_dataset_dir = dataset_dir + "train/"
test_dataset_dir = dataset_dir + "test_stg1/"


image_size = (512, 512)
classifier_input_size = (299, 299)


# class name
class_names_to_int = {
    'ALB': 0,
    'BET': 1,
    'DOL': 2,
    'LAG': 3,
    'NoF': 4,
    'OTHER': 5,
    'SHARK': 6,
    'YFT': 7,
}