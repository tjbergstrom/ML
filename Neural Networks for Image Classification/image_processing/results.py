# Ty Bergstrom
# Monica Heim
# results.py
# CSCE A415
# April 23, 2020
# ML Final Project
# display metrics and plots for any model

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras import metrics
import numpy as np
import matplotlib
matplotlib.use("Agg")

class Result:
	def display_metrix(testX, testY, predictions, model, classes, aug, bs):
		cl = Result.clas_report(testY, predictions, classes)
		cm = Result.confusion(model, aug, testX, testY, predictions)
		print("...classification report\n")
		print(cl)
		print("...confusion matrix\n")
		print(cm)
		print()
		Result.save_results(cl, cm)


	def clas_report(testY, predictions, classes):
		return classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classes)


	def confusion(model, aug, testX, testY, predictions):
		predIdxs = model.predict_generator(aug.flow(testX, testY))
		predIdxs = np.argmax(predIdxs, axis=1)
		return confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))


	def display_plot(plot, epochs, H):
		plt.style.use("ggplot")
		plt.figure()
		N = epochs
		plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
		plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
		plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
		plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
		plt.title("Training Loss and Accuracy")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.savefig(plot)
		plt.show()


	def save_results(cl, cm):
		f = open("image_processing/performance.txt","a+")
		f.write(cl)
		f.write("\n")
		f.write(np.array2string(cm))
		f.write("\n\n\n")


	def save_info(model, opt, aug, imgsz, epochs, bs, hxw):
		label = "model= {}, opt= {}, aug= {}, imgsz {}, epochs= {}, bs= {}, hxw= {}\n".format(model, opt, aug, imgsz, epochs, bs, hxw)
		f = open("image_processing/performance.txt","a+")
		f.write(label)
		f.write("\n")



