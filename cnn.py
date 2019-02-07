import glob
from keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D
from keras.backend import int_shape
from matplotlib import pyplot as plt
from numpy import array

from glob import glob

from scrapers import scrape

base_dir = './data'
datasets = ['mtb, city, road']


# scrape(0.1, rescrape=False)


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

input_layer = Input((64, 64, 1))
layer = Conv2D(1, (5, 5), activation='relu')(input_layer)
layer = MaxPool2D()(layer)
# print(int_shape(layer))
layer = Conv2D(1, (5, 5), activation='sigmoid')(layer)
layer = MaxPool2D()(layer)
# print(int_shape(layer))
layer = Conv2D(1, (5, 5), activation='relu')(layer)
layer = MaxPool2D()(layer)
# print(int_shape(layer))
layer = Conv2D(1, (4, 4), activation='sigmoid')(layer)
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



history = model.fit_generator(training_data, epochs=500, validation_data=validation_data)

# print(history.history.keys())

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Categorical accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.savefig('text/accuracy.svg')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.savefig('text/loss.svg')
plt.show()
