

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense

class CustomMLP:
	def preprocess(dataset):
		# not necessary
        return dataset

	def build(width, height, depth, classes):
		input_shape = (height, width, depth)
		model = Sequential()
		model.add(Dense(512, input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(Dense(128, activation="relu"))
		model.add(Dense(classes))
		model.add(activation=("softmax"))
		return model

	def optimize():
		opt = SGD(lr=0.1, decay=0.00001, nesterov=True)
		return opt

	def compile(model, opt):
		model.compile(loss='spare_categorical_crossentropy',
		optimizer=opt, metrics='accuracy')
		return model


