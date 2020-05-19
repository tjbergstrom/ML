# models.py
# these are the neural networks


from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers.core import Dense
from keras.layers import Concatenate
from keras.layers.core import Flatten
from keras.layers import Conv2DTranspose
from keras.layers.core import Activation
from keras.initializers import RandomNormal
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


# net for the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_image = Input(shape=image_shape)

	d = Conv2D(16, (3,3), strides=(2,2), padding='same',
	kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(32, (3,3), strides=(2,2), padding='same',
	kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Dropout(0.5)(d)

	d = Conv2D(64, (3,3), strides=(2,2), padding='same',
	kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Dropout(0.5)(d)

	#d = Flatten()(d)
	#d = Dense(64)(d)
	#d = (Activation('relu'))(d)
	#d = Dropout(0.5)(d)

	patch_out = Conv2D(1, (3,3), padding='same', kernel_initializer=init)(d)
	model = Model(in_image, patch_out)
	#opt = Adam(lr=0.0002, beta_1=0.5)
	opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
	#opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True))
	model.compile(loss='mse', optimizer=opt, loss_weights=[0.5])
	return model


# residual layers for the generator model
def resnet_block(n_filters, input_layer):
	init = RandomNormal(stddev=0.02)

	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Dropout(0.5)(g)

	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)

	g = Concatenate()([g, input_layer])
	return g


# net for the generator model
def define_generator(image_shape, n_resnet=3):
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)

	g = Conv2D(16, (3,3), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)

	g = Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Dropout(0.5)(g)

	g = Conv2D(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Dropout(0.5)(g)

	for _ in range(n_resnet):
		g = resnet_block(64, g)

	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Dropout(0.5)(g)

	g = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Dropout(0.5)(g)

	g = Conv2D(3, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	model = Model(in_image, out_image)
	return model


def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	g_model_1.trainable = True
	d_model.trainable = False
	g_model_2.trainable = False

	# discriminator element
	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)

	# identity element
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)

	# forward cycle
	output_f = g_model_2(gen1_out)

	# backward cycle
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)

	model = Model([input_gen, input_id],
	[output_d, output_id, output_f, output_b])

	#opt = Adam(lr=0.0002, beta_1=0.5)
	opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
	#opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True))
	model.compile(loss=['mse', 'mae', 'mae', 'mae'],
	loss_weights=[1, 5, 10, 10], optimizer=opt)
	return model



