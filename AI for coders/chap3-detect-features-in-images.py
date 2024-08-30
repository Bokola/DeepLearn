# detect features with convolutional neural networks (CNN)
# a convolution is a filter of weights (a square grid) used to multiply
# a pixel with its neighbours to get a new value for the pixel
# pooling: eliminating pixels in your image while maintaining the
# semantics of the content within the image
# keras uses max pooling to reduce the image

import tensorflow as tf
<<<<<<< HEAD
import numpy as np

=======
>>>>>>> 718f597f3a83a4e00100055d65f6fef218453ffd
data = tf.keras.datasets.fashion_mnist
((train_images, train_labels),
 (test_images, test_labels)) = data.load_data()
# make images 3 dim despite being gray scale which is 2 dim
# take 1 for gray scale and 3 for color images
train_images = train_images.reshape(60000,28,28,1)
train_images = train_images / 255.0
test_images = test_images.reshape(10000,28,28,1)
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
#     3 X 3 64 neurons for learning
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu',
                            input_shape = (28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
#     flatten
    tf.keras.layers.Flatten(),
#     dense layer
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
#     output layer
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(train_images, train_labels,epochs=50)
model.evaluate(test_images, test_labels)
# predictions
cl = model.predict(test_images)
print(cl[0])
print(test_labels[0])
# explore the model
model.summary()
<<<<<<< HEAD

# using Keras ImageDataGenerator to work with unlabelled data

import urllib.request
import zipfile

url = "https://storage.googleapis.com/learning-datasets/horse-or-human.zip"
file_name = "horse-or-human.zip"
training_dir = "horse-or-human/training/"
urllib.request.urlretrieve(url, file_name)

zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()

# validation set

validation_url = "https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"
validation_file_name = "validation-horse-or-human.zip"
validation_dir = 'horse-or-human/validation/'

urllib.request.urlretrieve(validation_url, validation_file_name)
zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

# scale images by 1/255
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode = 'binary'
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode = 'binary'
)

# images are colored needing 3 channels instead of one
# images may be larger than 300 X 300 pixels needing more layers

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu',
                           input_shape = (300, 300, 3)),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(528, activation = 'relu'),
    # one neuron as it's a binary classifier
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
model.summary()
# train using binary cross entropy loss function
model.compile(loss = 'binary_crossentropy',
              optimizer = RMSprop(learning_rate = 1e-3),
              metrics = ['accuracy'])
# train
history = model.fit(train_generator, epochs = 15,
                    validation_data = validation_generator)

# test the model
from PIL import Image
import glob
import os
import pathlib
from keras.preprocessing import image
from PIL import Image
import scipy

data_dir = os.path.join(os.getcwd(), 'data')
images = []
for f in glob.glob(os.path.join(data_dir, '*.jpg')):
    imm = Image.open(f)
    # comply to 300 X 300 size of the trained model
    img = image.load_img(f, target_size=(300, 300))
    # img.show()
    im = image.img_to_array(img)
    # make 3D
    im = np.expand_dims(im, axis=0)
    # stack vertically to match training data
    im_tensor = np.vstack([im])
    images.append(imm)
    # model overfits towards horses if human isn't fully posed
    classes = model.predict(im_tensor)
    print(classes)
    print(classes[0])
    if classes[0] > 0.5:
        print(f + ' is a human')
    else:
        print(f + ' is a horse')
# image augmentation: extend your training set beyond what is locally available to your date
# e.g transformations like color scheme, brightness levels, rotations, flipping, etc
# training takes longer coz of all the image preprocessing but improves predictions

train_datagen = ImageDataGenerator(
    rescale=1./255,
#     rotate randomly up to 40 degrees left/right
    rotation_range=40,
#     translate up to 20% vertically/horizontally
    width_shift_range=0.2,
    height_shift_range=0.2,
#     shearing up to 20%
    shear_range=0.2,
#     zoom up to 20%
    zoom_range=0.2,
#     random flip vertically/horizontally
    horizontal_flip=True,
#     fill any missing pixel after a move/shear with nearest neighbours
    fill_mode = 'nearest'
)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode = 'binary'
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode = 'binary'
)

# images are colored needing 3 channels instead of one
# images may be larger than 300 X 300 pixels needing more layers

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu',
                           input_shape = (300, 300, 3)),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(528, activation = 'relu'),
    # one neuron as it's a binary classifier
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
model.summary()
# train using binary cross entropy loss function
model.compile(loss = 'binary_crossentropy',
              optimizer = RMSprop(learning_rate = 1e-3),
              metrics = ['accuracy'])
# train
history = model.fit(train_generator, epochs = 15,
                    validation_data = validation_generator)

# test the model after augmentation

data_dir = os.path.join(os.getcwd(), 'data')
images = []
for f in glob.glob(os.path.join(data_dir, '*.jpg')):
    imm = Image.open(f)
    # comply to 300 X 300 size of the trained model
    img = image.load_img(f, target_size=(300, 300))
    # img.show()
    im = image.img_to_array(img)
    # make 3D
    im = np.expand_dims(im, axis=0)
    # stack vertically to match training data
    im_tensor = np.vstack([im])
    images.append(imm)
    # model overfits towards horses if human isn't fully posed
    classes = model.predict(im_tensor)
    print(classes)
    print(classes[0])
    if classes[0] > 0.5:
        print(f + ' is a human')
    else:
        print(f + ' is a horse')

# transfer learning: using prelearned layers from a larger model in our training
# use Google's inception model

from tensorflow.keras.applications.inception_v3 import InceptionV3
weights_url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)

pretrained_model = InceptionV3(input_shape = (150, 150, 3),
                               include_top = False,
                               weights = None)
pretrained_model.load_weights(weights_file)
# summary of pretrained model
pretrained_model.summary()
# set layers untrainable and point it to mixed7 as its output

