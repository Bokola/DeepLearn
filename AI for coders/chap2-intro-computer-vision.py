# computer vision: make a comp recognize objects
# an image is a rectangular grid of pixel, each pixel
# having a value btwn 0 to 255

# Classifying MNIST dataset

import tensorflow as tf
data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
# normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
#     input layer
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # dense layer with 128 neurons for learning
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     output layer
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
# for a classification, loss function is
# sparse_categorical_crossentropy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels)
# model output
cl = model.predict(test_images)
print(cl[0]) # probabilities to be in a category
print(test_labels[0]) # winning class

# stop training when desired accuracy reached using a callback
class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>=0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = mycallback()
mnist = tf.keras.datasets.mnist
((train_images, train_labels),
 (test_images, test_labels)) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=50,
          callbacks=[callbacks])