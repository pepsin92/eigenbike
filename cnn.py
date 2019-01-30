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

from scrapers import scrape
# scrape(0.2)



def load_data(directory=base_dir, categories=None):
    def replace_color(arr):
        col = arr[0][0][0]
        # print(col, arr[arr==col])
        arr[arr == col] = 255
        return arr

    idg = ImageDataGenerator(preprocessing_function=replace_color)

    data = idg.flow_from_directory(directory, (64, 64), 'grayscale', class_mode='categorical')

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

training_data = load_data(base_dir+'/training')

print(len(training_data), training_data)

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
layer = Dense(training_data.num_classes, activation='softmax')(layer)
# print(int_shape(layer))

model = Model(input_layer, layer)

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['categorical_accuracy'])

model.summary()

validation_data = load_data(base_dir+'/validation')

model.fit_generator(training_data, epochs=100, validation_data=validation_data)
