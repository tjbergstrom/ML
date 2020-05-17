# classify.py
# input an image
# detect any faces in the image
# compare them to faces that the model has been trained to recognize
# output the result

# run with:
# python3 classify.py -i inputs/1.jpg

import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,)
ap.add_argument("-c", "--confidence", type=float, default=0.5)
args = vars(ap.parse_args())

protoPath = "data/deploy.prototxt"
modelPath = "data/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch("data/openface_nn4.small2.v1.t7")
recognizer = pickle.loads(open("data/recognizer.pickle", "rb").read())
le = pickle.loads(open("data/le.pickle", "rb").read())

image = cv2.imread(args["image"])
#image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]
imageBlob = cv2.dnn.blobFromImage(
cv2.resize(image, (300, 300)), 1.0, (300, 300),
(104.0, 177.0, 123.0), swapRB=False, crop=False)
detector.setInput(imageBlob)
detections = detector.forward()

# loop over any detected faces in the input image
for i in range(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2]
	if confidence > args["confidence"]:
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]
		if fW < 20 or fH < 20:
			continue

		# quantify the detected face
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
		(96, 96), (0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# recognize the face - compare the quantification to the embeddings
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		# set up the output
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
#save_img = args["image"] + "_detected.jpg"
cv2.imwrite(save_img, image)



