# tensorflow quick intro
import tensorflow as tf
print('Tensorflow version: ', tf.__version__)
# load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Build ML model: here a sequential model
# Sequential is useful for stacking layers where each layer has one input
# tensor and one output tensor

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
# model returns a vector of logits/log-odds scores
pred = model(x_train[:1]).numpy()
pred
# tf.nn.softmax() converts logits to probabilities

tf.nn.softmax(pred).numpy()
# define loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
loss_fn(y_train[:1], pred).numpy
# before building, configure and compile model using keras
model.compile(optimizer = "adam",
              loss = loss_fn,
              metrics = ['accuracy'])
# train model
model.fit(x_train, y_train, epochs = 5)
# evaluate
model.evaluate(x_test, y_test, verbose = 2)
# return probabilities
prob_m = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
prob_m(x_test[:5])


