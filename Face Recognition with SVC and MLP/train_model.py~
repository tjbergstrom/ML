
# python3 train_model.py

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.svm import SVC
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", type=str, default="data/embeddings.pickle")
ap.add_argument("-r", "--recognizer", type=str, default="data/recognizer.pickle")
ap.add_argument("-l", "--le", type=str, default="data/le.pickle")
ap.add_argument("-m", "--model", type=str, default="svc")
args = vars(ap.parse_args())

if (args["model"]).lower() == "svc":
    model = "SVC"
elif (args["model"]).lower() == "mlp":
    model = "MLP"
else:
    model = "SVC"

data = pickle.loads(open(args["embeddings"], "rb").read())
le = LabelEncoder()
labels = le.fit_transform(data["names"])
(trainX, testX, trainY, testY) = train_test_split(data["embeddings"], labels, test_size=0.25)

print(" training model...")
if model == "svc":
    model = SVC(C=1.0, kernel="linear", probability=True)
elif model == "mlp":
    model= MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation='relu', solver='adam', random_state=1)
model.fit(trainX, trainY)
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=le.classes_))
model.fit(data["embeddings"], labels)

f = open(args["recognizer"], "wb")
f.write(pickle.dumps(model))
f.close()
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
print((args["model"]), "model trained\n")



