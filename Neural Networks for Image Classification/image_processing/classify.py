# Ty Bergstrom
# Monica Heim
# classify.py
# CSCE A415
# April 23, 2020
# ML Final Project
# testing a model's accuracy

# run with this command
# (open the virtual env first)
# python3 -W ignore classify.py --image "dataset/gun/sherlock_gun.jpg"
# python3 -W ignore classify.py --image "dataset/phone/Liam-Neeson-Taken-Phone.jpg"

# this file takes an input image and model and classifies the image
# it ouputs whether the predicted classification was correct

from keras.preprocessing.image import img_to_array
from image_processing.tuning import Tune
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="image_processing/model.model")
ap.add_argument("-l", "--lb", type=str, default="image_processing/lb.pickle")
ap.add_argument("-s", "--imgsz", type=str, default="s")
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())

HXW = Tune.img_size(args["imgsz"])
image = cv2.imread(args["image"])
output = image.copy()
image = cv2.resize(image, (HXW, HXW))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
model = load_model(args["model"])
lb = pickle.loads(open(args["lb"], "rb").read())

prob = model.predict(image)[0]
idx = np.argmax(prob)
label = lb.classes_[idx]
filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
if filename.rfind(label) != -1:
    guess = "correct"
else:
    guess = "incorrect"
label = "{}: {:.2f}%".format(label, prob[idx] * 100)
output = imutils.resize(output, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

print("...the model's guess:", lb.classes_[idx], "with {:.2f}% accuracy ({})".format(prob[idx] * 100, guess))
cv2.imshow("Output", output)
cv2.imwrite(filename, output)
cv2.waitKey(0)



