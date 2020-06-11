# Ty Bergstrom
# Monica Heim
# lenet.py
# CSCE A415
# April 23, 2020
# ML Final Project
# build a LeNet neural network


from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
	@staticmethod
	def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# first set of convolutional relu and pooling layers
		model.add(Conv2D(32, (5, 5), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# second set of convolutional relu and pooling layers
		model.add(Conv2D(64, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# only set of fully connected relu layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model


