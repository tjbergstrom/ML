# Ty Bergstrom
# Monica Heim
# cnn_model.py
# CSCE A415
# April 23, 2020
# ML Final Project
# build a convolutional neural network


from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as K

class CNNmodel:
	@staticmethod
	def build(width, height, depth, classes):
		inputShape = (height, width, depth)
		model = Sequential()

		# convolutional layer
		model.add(Conv2D(8, (3, 3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# another convolutional layer
		model.add(Conv2D(16, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# another convolutional layer
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Flatten())

		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model


