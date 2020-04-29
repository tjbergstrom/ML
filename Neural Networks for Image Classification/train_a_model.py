# Ty Bergstrom
# Monica Heim
# train_a_model.py
# CSCE A415
# April 23, 2020
# ML Final Project
# train any model with this file
#
# terminal command usage:
# open the virtual environment:
# $ source ./venv1/bin/activate
# and run with this command:
# $ python3 -W ignore train_a_model.py
#
# this file pre-processes an input dataset with selected parameters
# trains a selected model with selected parameters and tunings and saves it
# outputs metrics and saves them, makes a plot and saves it


from sklearn.preprocessing import LabelBinarizer
from image_processing.process import Pprocess
from image_processing.results import Result
from image_processing.tuning import Tune
from deepnet.dnet import Deepnet
from cnn.cnn_net import CNNmodel
from vgg.vggnet import VGGmodel
from lenet.lenet import LeNet
import numpy as np
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--savemodel", type=str, default="image_processing/model.model")
ap.add_argument("-p", "--plot", type=str, default="image_processing/plot.png")
ap.add_argument("-d", "--dataset", type=str, default="newdataset2")
ap.add_argument("-a", "--aug", type=str, default="original")
ap.add_argument("-m", "--model", type=str, default="lenet")
ap.add_argument("-o", "--opt", type=str, default="Adam2")
ap.add_argument("-i", "--imgsz", type=str, default="m")
ap.add_argument("-e", "--epochs", type=int, default=50)
ap.add_argument("-b", "--bs", type=str, default="m")
args = vars(ap.parse_args())

EPOCHS = args["epochs"]
BS = Tune.batch_size(args["bs"])
HXW = Tune.img_size(args["imgsz"])

print("\n...pre-processing the data...\n")
(data, cl_labels) = Pprocess.preprocess(args["dataset"], HXW)
lb = LabelBinarizer()
cl_labels = lb.fit_transform(cl_labels)
num_classes = len(lb.classes_)
loss_type = "binary_crossentropy"
if num_classes > 2:
    loss_type = "categorical_crossentropy"
(trainX, testX, trainY, testY) = Pprocess.split(data, np.array(cl_labels), num_classes)
aug = Pprocess.dataug(args["aug"])

print("\n...building the model...\n")
if args["model"] == "lenet":
    model = LeNet.build(width=HXW, height=HXW, depth=3, classes=num_classes)
if args["model"] == "cnn":
    model = CNNmodel.build(width=HXW, height=HXW, depth=3, classes=num_classes)
if args["model"] == "vgg":
    model = VGGmodel.build(width=HXW, height=HXW, depth=3, classes=num_classes)
if args["model"] == "deepnet":
    model = Deepnet.build(width=HXW, height=HXW, depth=3, classes=num_classes)
opt = Tune.optimizer(args["opt"], EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("\n...training the model...\n")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)
model.save(args["savemodel"] )
f = open("image_processing/lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()

print("\n...getting results of training & testing...\n")
Result.save_info(args["model"], args["opt"], args["aug"], args["imgsz"], EPOCHS, BS, HXW)
predictions = model.predict(testX, batch_size=BS)
Result.display_metrix(testX, testY, predictions, model, lb.classes_, aug, BS)
Result.display_plot((args["plot"]), EPOCHS, H)



