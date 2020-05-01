
# python3 process.py

from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--embedding-model", type=str, default="openface_nn4.small2.v1.t7")
ap.add_argument("-e", "--embeddings", type=str, default="data/embeddings.pickle")
ap.add_argument("-d", "--detector", type=str, default="face_detection_model")
ap.add_argument("-i", "--dataset", type=str, default="dataset")
ap.add_argument("-c", "--confidence", type=float, default=0.5)
args = vars(ap.parse_args())

protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

imagePaths = list(paths.list_images(args["dataset"]))
knownEmbeddings = []
knownNames = []
total_faces = 0

for (i, imagePath) in enumerate(imagePaths):
	print(" processing image {}/{}".format(i + 1, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# detect faces in the image
	imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
	detector.setInput(imageBlob)
	detections = detector.forward()

	if len(detections) < 1:
		break
	else:
		# get the face that was detected with highest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
			if fW < 20 or fH < 20:
				continue

			# quantify the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total_faces += 1

data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()
print("Data saved\n")



