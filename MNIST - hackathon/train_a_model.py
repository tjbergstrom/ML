# Ty Bergstrom
# train_a_model.py
# June 25, 2020
# Identify the Digits (MNIST)
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
#
# python3 -W ignore train_a_model.py -e 10 -c train2.csv -m Nnet2 -d images/train/


from sklearn.preprocessing import LabelBinarizer
from image_processing.process import Pprocess
from image_processing.results import Result
from image_processing.tuning import Tune
from NeuralNet.net import Nnet1
from NeuralNet.net import Nnet2
import numpy as np
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--savemodel", type=str, default="image_processing/model.model")
ap.add_argument("-p", "--plot", type=str, default="image_processing/plot.png")
ap.add_argument("-d", "--dataset", type=str, default="images")
ap.add_argument("-a", "--aug", type=str, default="original")
ap.add_argument("-m", "--model", type=str, default="Nnet1")
ap.add_argument("-k", "--kernelsize", type=int, default=3)
ap.add_argument("-o", "--opt", type=str, default="Adam2")
ap.add_argument("-i", "--imgsz", type=str, default="xs")
ap.add_argument("-e", "--epochs", type=int, default=50)
ap.add_argument("-b", "--bs", type=str, default="m")
ap.add_argument("-c", "--cvfile", type=str, default="train.csv")
args = vars(ap.parse_args())

EPOCHS = args["epochs"]
BS = Tune.batch_size(args["bs"])
HXW = Tune.img_size(args["imgsz"])
k = args["kernelsize"]

print("\n...pre-processing the data...\n")
(data, cl_labels) = Pprocess.preprocess(args["dataset"], args["cvfile"], HXW)
lb = LabelBinarizer()
cl_labels = lb.fit_transform(cl_labels)
num_classes = len(lb.classes_)
loss_type = "binary_crossentropy"
if num_classes > 2:
    loss_type = "categorical_crossentropy"
(trainX, testX, trainY, testY) = Pprocess.split(data, np.array(cl_labels), num_classes)
aug = Pprocess.dataug(args["aug"])

print("\n...building the model...\n")
if args["model"] == "Nnet1":
    model = Nnet1.build(width=HXW, height=HXW, depth=3, k=k, classes=num_classes)
if args["model"] == "Nnet2":
    model = Nnet2.build(width=HXW, height=HXW, depth=3, k=k, classes=num_classes)
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



