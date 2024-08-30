# saving models in HDF5 formats for keras
# saving models in TensorFlow SavedModel format
# loading models
# Download model to local disk

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras import layers

# load data
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)
# find zip dir
zip_dir_base = os.path.dirname( zip_dir)
base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# understanding our data
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

BATCH_SIZE = 100
IMAGE_SIZE = 150

train_image_generator      = ImageDataGenerator(rescale=1./255)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(
   directory=train_dir,
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode='binary'
)

val_data_gen = validation_image_generator.flow_from_directory(
    directory=validation_dir,
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode='binary'
)
# (train, validate), info = tfds.load(
#     'cats_vs_dogs',
#     split=['train[:80%]', 'validate[80%:]'],
#     with_info=True,
#     as_supervised=True,
# )

# transfer learning with Tensorflow

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(
    URL,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE,3))

# freeze the variables in feature extractor layer
feature_extractor.trainable = False
# attach a classification layer
model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(2)
])