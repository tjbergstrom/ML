# process.py
# input original images
# detect and extract a face from an image
# save this face as a new 256x256 image

# this is important because the models need good photos of just faces
# the face detector is pretty accurate, but check the processed dataset afterwords and delete any photos that aren't actually a face

# python3 process.py -d A_girl_selfies
# python3 process.py -d A_guy_selfies
# python3 process.py -d B_girl_anime
# python3 process.py -d B_guy_anime

from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-c", "--confidence", type=float, default=0.8)
args = vars(ap.parse_args())

protoPath = "face_detector/deploy.prototxt"
modelPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
dset = args["dataset"]
imagePaths = list(paths.list_images("raw_dataset/" + dset))

for (itr, imagePath) in enumerate(imagePaths):
	#print(imagePath)
	#print(" processing image {}/{}".format(itr+1, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	#image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# detect faces in the image
	imageBlob = cv2.dnn.blobFromImage(cv2.resize(image,
	(300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
	swapRB=False, crop=False)
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
			if startY - 32 > 0:
				startY -= 32
			if startX - 32 > 0:
				startX -= 32
			face = image[startY:endY+32, startX:endX+32]
			(fH, fW) = face.shape[:2]
			if fW < 64 or fH < 64:
				continue
			face = imutils.resize(face, width=256, height=256)
			#cv2.imshow("face",face)
			#cv2.waitKey(0)
			filename = "processed_dataset/" + dset + "/" + str(itr+1) + ".jpg"
			cv2.imwrite(filename, face)



