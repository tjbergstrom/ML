
from sklearn.svm import SVC

class DefaultSVC:
	def build(x, y, tune):
		gamma = gamma_tuning(0)
		degree = degree_tuning(0) # only for poly kernel
		c = c_tuning(0)
		kernel = kerel_tuning(0)
		model = SVC(C=c, kernel=kernel, gamma=gamma, degree=degree)
		model.fit(x, y)
		return model

	def gamma_tuning(tune):
		gammas = [0.1, 1, 10, 100]
		return gammas[tune]

	def degree_tuning(tune):
		degrees = [0, 1, 2, 3, 4, 5, 6]
		return degrees[tune]

	def c_tuning(tune):
		cs = [0.1, 1, 10, 100, 1000]
		return cs[tune]

	def kerel_tuning(tune):
		kernels = ["linear", "rbf", "poly"]
		return kernels[tune]

	def predict(model, trainX, testX):
		model.predict(self.gaussian_matrix(testX, trainX))
		return model



