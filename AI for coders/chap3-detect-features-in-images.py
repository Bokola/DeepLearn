# detect features with convolutional neural networks (CNN)
# a convolution is a filter of weights (a square grid) used to multiply
# a pixel with its neighbours to get a new value for the pixel
# pooling: eliminating pixels in your image while maintaining the
# semantics of the content within the image
# keras uses max pooling to reduce the image

import tensorflow as tf
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
