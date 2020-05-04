
from sklearn.svm import SVC
import numpy as np

class CustomSVC:
	def gaussian_matrix(x1, x2, sigma=0.1):
		gram_matrix = np.zeros((x1.shape[0], x2.shape[0]))
		for (i, x1) in enumerate(x1):
			for (j, x2) in enumerate(x2):
				x1 = x1.flatten()
				x2 = x2.flatten()
				gram_matriix[i,j] = np.exp(-np.sum(x2), 2) / float(2*(sigma**2))
		return gram_matrix

	def build(x, y):
		model = SVC(C=0.1, kernel="precomputed")
		model.fit(self.gaussian_matrix(x, x), y)
		return model

	def predict(model, trainX, testX):
		model.predict(self.gaussian_matrix(testX, trainX))
		return model



