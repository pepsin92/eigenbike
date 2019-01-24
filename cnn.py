import glob
from keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Dense, Flatten
from keras.backend import int_shape
from matplotlib import pyplot as plt
from numpy import array

from glob import glob

base_dir = './data'
datasets = ['mtb, city, road']


def load_data(directory=base_dir, categories=None):
    def replace_color(arr):
        col = arr[0][0][0]
        # print(col, arr[arr==col])
        arr[arr == col] = 255
        return arr

    idg = ImageDataGenerator(preprocessing_function=replace_color)

    data = idg.flow_from_directory(base_dir, (64, 64), 'grayscale', class_mode='categorical')

    return data

# example = 'data/road/0.jpeg'
#
# img = load_img(example).convert('L')
# print(img)
#
# X_input = img_to_array(img)
#
# print(X_input.shape)
#
# X_input = []
# Y_input = []

data = load_data()

print(len(data), data)

# for x, y in data:
#     print(x.shape, y.shape)
#     break
    # foo = array_to_img(x[0])
    # print(foo)
    # foo.show()
    # exit(0)

# MODEL

# model = Sequential()
# model.add(InputLayer((64, 64, 1)))
# model.add(Conv2D(4, (5, 5), activation='sigmoid'))

input_layer = Input((64, 64, 1))
layer = Conv2D(1, (5, 5), strides=(2, 2), activation='relu')(input_layer)
# print(int_shape(layer))
layer = Conv2D(1, (5, 5), strides=(2, 2), activation='sigmoid')(layer)
# print(int_shape(layer))
layer = Conv2D(1, (5, 5), strides=(2, 2), activation='relu')(layer)
# print(int_shape(layer))
layer = Conv2D(1, (5, 5), strides=(2, 2), activation='sigmoid')(layer)
# print(int_shape(layer))
layer = Flatten()(layer)
# print(int_shape(layer))
# layer = Dense(4, activation='relu')(layer)
layer = Dense(3, activation='softmax')(layer)
# print(int_shape(layer))

model = Model(input_layer, layer)

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['categorical_accuracy'])

model.summary()

model.fit_generator(data, epochs=1000)
