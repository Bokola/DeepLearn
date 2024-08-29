import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# define a layer using Sequential
# here 1 dense layer with a one-dim array as shape
model = Sequential([
    Dense(units=1, input_shape=[1])
])
# stochastic gradient descent (sdg) optimizer for numerical problems
model.compile(optimizer='SGD', loss='mean_squared_error')
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

model.fit(xs, ys, epochs=500)
print(model.predict(np.array([10.0])))
# y = wx + b (w = weight, b = bias)
print('This is what I learned: {}'.format(model.get_weights()))