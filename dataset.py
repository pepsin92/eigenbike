from keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from numpy import array


def load_data(directory, categories=None, as_array=False):
    def replace_color(arr):
        col = arr[0][0][0]
        # print(col, arr[arr==col])
        arr[arr == col] = 255
        return arr

    idg = ImageDataGenerator(preprocessing_function=replace_color)

    data = idg.flow_from_directory(directory, (64, 64), 'grayscale', class_mode='categorical')

    if not as_array:
        return data

    X_inputs = []
    Y_inputs = []

    while True:
        a = next(data)
        X_inputs.extend(a[0])
        Y_inputs.extend(a[1])
        # print(data.batch_index)
        if data.batch_index == 0:
            break

    return array(X_inputs), array(Y_inputs), data
