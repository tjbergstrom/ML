# test the model with un-labeled images
# make predicions with the model
# and save them to a file to submit

# python3 -W ignore test.py | tee test3.csv


from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2
import csv

dataset = "images/test/"
testfile = "Test_fCbTej3_0j1gHmj.csv"
testwrite = "test2.csv"
model = load_model("image_processing/model.model")
lb = pickle.loads(open("image_processing/lb.pickle", "rb").read())
HXW = 24
#labels = []
#img_paths = []
img_labels = []

with open(testfile, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for line in csv_reader:
        imagePath = dataset + line[0]
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (HXW, HXW))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]
        #labels.append(labels)
        #img_paths.append(line[0])
        img_label = line[0] + "," + label
        img_labels.append(img_label)
        print(img_label)

with open(testwrite, "w") as csv_file:
    for line in img_labels:
        csv_file.write(line)
        csv_file.write("\n")




