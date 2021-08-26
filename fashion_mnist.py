import struct

import numpy as np
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

import config as cfg


def preprocess_data(x_train, x_test, img_rows, img_cols):
    
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.0
    x_test /= 255.0

    return x_train, x_test


def read_idx(filename):
    """Credit: https://gist.github.com/tylerneylon"""
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

base_dir = "C:/Users/himan/Documents/cv_code/Code_TensorFlow2_Version/13. Building LeNet and AlexNet in Keras/fashion"
x_train = read_idx(base_dir + "/train-images-idx3-ubyte")
y_train = read_idx(base_dir + "/train-labels-idx1-ubyte")
x_test = read_idx(base_dir + "/t10k-images-idx3-ubyte")
y_test = read_idx(base_dir + "/t10k-labels-idx1-ubyte")

batch_size = cfg.batch_size
epochs = cfg.epochs

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

x_train, x_test = preprocess_data(x_train, x_test, img_rows, img_cols)

input_shape = (img_rows, img_cols, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]


#model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = SGD(0.01),
              metrics = ['accuracy'])

print(model.summary())
