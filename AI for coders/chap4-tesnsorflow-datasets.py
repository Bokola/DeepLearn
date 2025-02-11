import tensorflow as tf
import tensorflow_datasets as tfds
import datasets

# data from {datasets}

from datasets import load_dataset

# Print all the available datasets
from huggingface_hub import list_datasets
print([dataset.id for dataset in list_datasets()])

mnist_data = tfds.load('fashion_mnist')
for item in mnist_data:
    print(item)

# specify a split of interest
mnist_train = tfds.load('fashion_mnist', split='train')
print(type(mnist_train))
# type of item in each record
for item in mnist_train.take(1):
    print(item)
    print(item.keys())
    print(item['image'])
    print(item['label'])

# data from tfds
data = tfds.load('horses_or_humans', split='train', as_supervised=True)
# for tfds you have to batch your data
# batch in 100s
train_batches = data.shuffle(100).batch(10)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_batches, epochs = 10)
# batch validation set as well
val_data = tfds.load('horses_or_humans', split='test', as_supervised=True)
validation_batches = val_data.batch(32)
history = model.fit(train_batches, epochs = 10, validation_data = validation_batches)

# parallelize to improve training performance

train_data = tfds.load('horses_or_humans', split='train', as_supervised=True)
