import glob
from keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D
from keras.backend import int_shape
from keras.utils import plot_model
from matplotlib import pyplot as plt
from numpy import array
from time import process_time

from glob import glob

from scrapers import scrape

base_dir = './data'
datasets = ['mtb, city, road']

# UNCOMMENT TO NOT DOWNLOAD IMAGES
# scrape(0.1, rescrape=True)


def load_data(directory=base_dir, categories=None, save=None):
    def replace_color(arr):
        col = arr[0][0][0]
        # print(col, arr[arr==col])
        arr[arr == col] = 255
        return arr

    idg = ImageDataGenerator(preprocessing_function=replace_color)

    data = idg.flow_from_directory(directory, (64, 64), 'grayscale', class_mode='categorical', save_to_dir=save)

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

print('Loading training dataset')
training_data = load_data(base_dir+'/training')

print('Loading validation dataset')
validation_data = load_data(base_dir+'/validation')

# print(len(training_data), training_data)

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

stride = (2, 2)

input_layer = Input((64, 64, 1))
layer = Conv2D(2, (5, 5), strides=stride, activation='relu')(input_layer)
# layer = MaxPool2D()(layer)
# print(int_shape(layer))
layer = Conv2D(4, (5, 5), strides=stride, activation='sigmoid')(layer)
# layer = MaxPool2D()(layer)
# print(int_shape(layer))
layer = Conv2D(8, (5, 5), strides=stride, activation='relu')(layer)
# layer = MaxPool2D()(layer)
# print(int_shape(layer))
layer = Conv2D(16, (5, 5), strides=stride, activation='sigmoid')(layer)
# print(int_shape(layer))
layer = Flatten()(layer)
# print(int_shape(layer))
# layer = Dense(4, activation='relu')(layer)
layer = Dense(training_data.num_classes, activation='softmax')(layer)
# print(int_shape(layer))

model = Model(input_layer, layer)

plot_model(model, 'foo.png')
# exit(0)

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['categorical_accuracy'])

model.summary()


start_time = process_time()
history = model.fit_generator(training_data, epochs=150, validation_data=validation_data)
end_time = process_time()

print(f"Time elapsed: {end_time-start_time:.3f} seconds")

# print(history.history.keys())

# print(history.history.items())

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Stride: Model accuracy')
plt.ylabel('Categorical accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
# plt.savefig('text/stride_accuracy.png')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Stride: Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
# plt.savefig('text/stride_loss.png')
plt.show()
