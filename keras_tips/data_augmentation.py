from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from matplotlib import pyplot
import os
from keras import backend as K

K.set_image_dim_ordering('th')
""" 
вдохновение:  http://machinelearningmastery.com/image-augmentation-deep-learning-keras/
документация: https://keras.io/preprocessing/image/
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
# convert from int to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

model = Sequential()
""" некая модель здесь """


def with_standartization():
  return ImageDataGenerator(featurewise_center=True,
                            featurewise_std_normalization=True)


def with_shifting_and_flipping():
  return ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
  )


def with_whitening():
  return ImageDataGenerator(zca_whitening=True)


def with_rotation():
  return ImageDataGenerator(rotation_range=90)


datagen_train = with_shifting_and_flipping()
datagen_train.fit(x_train)

os.makedirs('images')
"""
save_to_dir - сохраняем в директорию
"""
for x_batch, y_batch in datagen_train.flow(x_train, y_train, batch_size=9,
                                           save_to_dir='images', save_prefix='aug',
                                           save_format='png'):
  # create a grid of 3x3 images
  for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
  # show the plot
  pyplot.show()
  break

""" Вместо fit используем fin_generator и передаем в него flow """
model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=x_train.shape[0],
                    epochs=10, verbose=2)
