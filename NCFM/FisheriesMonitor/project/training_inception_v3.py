import numpy as np
import settings
from data_managers import load_train_dataset
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from myutils import preprocessing_for_ipv3, create_inception_v3


# fix random seed
seed = 7
np.random.seed(seed)
num_classes = 8
learning_rate = 0.0001


# setting variables
img_size = (299, 299)
inits_weights = '../models/ipv3_ncfm_t50.h5'
model_filename = '../models/ipv3_ncfm_best.h5'
nb_epoch = 64
batch_size = 32
learning_rate = 0.0001


# build ip_v3 and init_weights
ip_v3 = create_inception_v3(img_size)
ip_v3.load_weights(inits_weights)



print "Loading Data..."
X, Y = load_train_dataset(verbose=0, img_size=img_size) # load dataset
X = preprocessing_for_ipv3(X) # preprocess
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=seed) # split dataset

# trans Y_train to one-hot-vectory
Y_train_vec = to_categorical(Y_train)
Y_val_vec = to_categorical(Y_val)
print "Fish loading Data!"


# Data augumentation
train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
)
train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, Y_train_vec, batch_size=batch_size)


# training model
print "Start Training Model ..."
checkpoint = ModelCheckpoint(model_filename, monitor='val_loss', verbose = 1, save_best_only = True)
ip_v3.fit_generator(
    train_generator,
    samples_per_epoch=len(X_train), 
    validation_data=(X_val, Y_val_vec),
    nb_epoch=nb_epoch,
    callbacks=[checkpoint],
    verbose=1
)


# load model and evaluate model
print "Evaluate Model"
trained_weights_url = model_filename
ip_v3.load_weights(trained_weights_url)
# evaluate mode
scores = ip_v3.evaluate(X_val, Y_val_vec, batch_size=batch_size, verbose=1)
print("val_loss: %s,  val_acc: %s"%(scores[0], scores[1]))


print "Training Complete:", model_filename
