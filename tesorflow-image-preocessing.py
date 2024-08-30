# run in console: Alt + Shift + E
# run all: Ctrl + Enter
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib

print(tf.__version__)
# download flower dataset

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path(archive).with_suffix('')
# count images
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
# roses
roses = list(data_dir.glob('roses/*'))
Image.open(roses[0])

# load data using keras
batch_size = 32
img_height = 180
img_width = 180
# train, test sets
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
# class names
class_names = train_ds.class_names
print(class_names)
# visualize the data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis('off')

# standardize data for neural networks
normalize_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalize_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# configure dataset for performance
# dataset.cache() keeps the images in memory after loading
# dataset.prefetch() overlaps data preprocessing & modelling while training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# train a CNN model
#  composed of convolutions and maxpooling
# convolutional layer applies a kernel (filter) over the regions of the
# input image
#  zero padding to cells on the edges
# Sequential model with 3 convolution blocks
# maxpooling: reducing the size of an input image by summarizing regions
# select a grid and select pixel with max value from that grid
# next select a stride: the no. of windows to slide the pixel across the image
# and again select the pixel with max value
# add the pixel and repeat throughout the size of the input image
# the result is a new image that is smaller than the original image
# fix overfitting: image augmentation & dropouts
# dropouts: randomly turning off some neurons during training to play
# around with weights. this mitigates overfitting

num_classes = len(class_names)
model = tf.keras.Sequential([
    # resize to same size
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])
model.compile(
    optimizer='adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.fit(train_ds, validation_data=val_ds, epochs=3)