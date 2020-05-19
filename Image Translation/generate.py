# generate.py
# upload an image a return the transation

# python3 -W ignore generate.py -i samples/2.jpg -m g_model_AtoB_005205.h5


import argparse
from numpy import load
from matplotlib import pyplot
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True)
ap.add_argument("-m", "--model", required = True)
args = vars(ap.parse_args())
HXW = 64


def load_image(filename, size=(HXW,HXW)):
	pixels = load_img(filename, target_size=size)
	pixels = img_to_array(pixels)
	pixels = expand_dims(pixels, 0)
	pixels = (pixels - 127.5) / 127.5
	return pixels


image = load_image(args["image"])
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model(args["model"], cust)
# translate image
image_tar = model_AtoB.predict(image)
# scale back from [-1,1] to [0,1]
image_tar = (image_tar + 1) / 2.0
pyplot.imshow(image_tar[0])
pyplot.show()



