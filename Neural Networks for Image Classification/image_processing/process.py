# Ty Bergstrom
# Monica Heim
# process.py
# CSCE A415
# April 23, 2020
# ML Final Project
# pre-process the dataset for any model


from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
import os

class Pprocess:
	# load the input dataset and process
	def preprocess(dataset, HXW):
		random.seed(42)
		data = []
		labels = []
		cl_labels = []
		imagePaths = sorted(list(paths.list_images(dataset)))
		random.seed(42)
		random.shuffle(imagePaths)
		for imagePath in imagePaths:
			image = cv2.imread(imagePath)
			image = cv2.resize(image, (HXW, HXW))
			image = img_to_array(image)
			data.append(image)
			label = imagePath.split(os.path.sep)[-2]
			cl_labels.append(label)
			if label == "gun":
				label = 1
			else:
				label = 0
			labels.append(label)
		data = np.array(data, dtype="float") / 255.0
		labels = np.array(labels)
		return data, labels, cl_labels


	def split(data, labels):
		(trainX, testX, trainY, testY) = train_test_split(data,
		labels, test_size=0.20, random_state=42)
		trainY = to_categorical(trainY, num_classes=2)
		testY = to_categorical(testY, num_classes=2)
		return trainX, testX, trainY, testY


	# different options for augmentation pre-processing
	def dataug(aug):
		if aug == "original":
			return ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
			horizontal_flip=True, fill_mode="nearest")
		elif aug == "light1":
			return ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
			height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
			horizontal_flip=True, fill_mode="nearest")
		elif aug == "light2":
			return ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
			height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
			horizontal_flip=True, fill_mode="reflect")
		elif aug == "light3":
			return ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
			height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
			horizontal_flip=True, fill_mode="wrap")
		elif aug == "medium1":
			return ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
			height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
			brightness_range=[1.0,1.5],	horizontal_flip=True, fill_mode="nearest")
		elif aug == "medium2":
			return ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
			height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
			brightness_range=[0.5,1.0],	horizontal_flip=True, fill_mode="nearest")
		elif aug == "medium3":
			return ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
			height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
			vertical_flip=True, fill_mode="nearest")
		elif aug == "heavy1":
			return ImageDataGenerator(rotation_range=45, width_shift_range=0.1,
			height_shift_range=0.1, shear_range=0.2, zoom_range=1.2,
			brightness_range=[1.0,1.0],	horizontal_flip=True, fill_mode="nearest")
		elif aug == "heavy2":
			return ImageDataGenerator(rotation_range=45, width_shift_range=0.1,
			height_shift_range=0.1, shear_range=0.2, zoom_range=1.2,
			brightness_range=[0.5,1.0],	horizontal_flip=True, fill_mode="nearest")
		else:
			return ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
			height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
			horizontal_flip=True, fill_mode="nearest")



