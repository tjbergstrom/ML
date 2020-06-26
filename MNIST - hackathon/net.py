# The neural networks


from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.layers.core import Dense
from keras import backend as K

class Nnet1:
	@staticmethod
	def build(width, height, depth, k, classes):
		model = Sequential()
		inputShape = (height, width, depth)
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# first set of convolutional relu and pooling layers
		model.add(Conv2D(32, (k, k), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.2))

		# second set of convolutional relu and pooling layers
		model.add(Conv2D(64, (k, k), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.2))

		# only set of fully connected relu layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model



class Nnet2:
	@staticmethod
	def build(width, height, k, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# first set of convolutional relu and pooling layers
		model.add(Conv2D(32, (k, k), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.2))

		# second set of convolutional relu and pooling layers
		model.add(Conv2D(64, (k, k), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.2))

		# third set of convolutional relu and pooling layers
		model.add(Conv2D(128, (k, k), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.2))

		# fourth set of convolutional relu and pooling layers
		model.add(Conv2D(264, (k, k), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.2))

		# only set of fully connected relu layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model



