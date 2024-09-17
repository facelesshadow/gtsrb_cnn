import sys
import os
import tensorflow as tf
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

NUM_CATEGORIES = 43
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
TEST_SIZE = 0.2


if len(sys.argv) != 2:
    sys.exit("Usage: python model.py data")


path = sys.argv[1]


label_list = os.listdir(path)
try:
    label_list.remove(".DS_Store")
except:
    ...

image_data = []
labels = []
for file1 in label_list:
    image_list = os.listdir(os.path.join(path, file1))
    for imagename in image_list:
        image_path = os.path.join(path, file1, imagename)
        image_array = cv2.imread(image_path)
        image_array = cv2.resize(image_array, (IMG_WIDTH, IMG_HEIGHT))
        image_data.append(image_array)
        labels.append(os.path.basename(file1))

labels = tf.keras.utils.to_categorical(labels)

#test-train split
x_train, x_test, y_train, y_test = train_test_split(
    np.array(image_data), np.array(labels), test_size = TEST_SIZE
)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32, (3,3), activation="relu", input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=EPOCHS)

model.evaluate(x_test, y_test)