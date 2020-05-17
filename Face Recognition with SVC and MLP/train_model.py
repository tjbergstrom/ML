


# python3 -W ignore train_model.py

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import accuracy_score
#from sklearn import cross_validation
from svc.othersvc import DefaultSVC
from svc.customsvc import CustomSVC
from mlp.custommlp import CustomMLP
from sklearn.svm import SVC
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mod", type=str, default="svc")
args = vars(ap.parse_args())

if (args["mod"]).lower() == "svc":
    mod = "SVC"
elif (args["mod"]).lower() == "mlp":
    mod = "MLP"
else:
    mod = "SVC"

data = pickle.loads(open(args["embeddings"], "rb").read())
le = LabelEncoder()
labels = le.fit_transform(data["names"])
(trainX, testX, trainY, testY) = train_test_split("data/embeddings.pickle", labels, test_size=0.25)

print(" training model...")
if mod == "SVC":
    model = SVC(C=1.0, kernel="poly", probability=True)
    #model = CustomSVC.build(trainX, trainY)
    #model = DefaultSVC.build(trainX, trainY, 0)
elif mod == "MLP":
    model = MLPClassifier(hidden_layer_sizes=(150,100,50),
    max_iter=300, activation='relu', solver='adam', random_state=1)
    #model = CustomMLP.build(96, 96, 3, 3)
    #opt = CustomMLP.optimize()
    #CustomMLP.compile(model, opt)
M = model.fit(trainX, trainY)
#predictions = model.predict(testX)
#print(classification_report(testY, predictions, target_names=le.classes_))
model.fit(data["embeddings"], labels)

f = open("data/recognizer.pickle", "wb")
f.write(pickle.dumps(model))
f.close()
f = open("data/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
print((args["mod"]), "model trained\n")



