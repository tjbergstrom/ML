
# python3 news.py -f news.csv

import argparse
import itertools
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", type=str, default="news.csv")
args = vars(ap.parse_args())
input_file = args["file"]

print("\n pre-processing data...\n")

df = pd.read_csv(input_file)
df.shape
labels = df.label
data = df['text']

data = data.str.lower()
sw = stopwords.words('english')
data = data.apply(lambda x: ' '.join([w for w in x.split() if w not in sw]))
data = data.str.replace('[^\w\s]','')
print(data.head())

trainX, testX, trainY, testY = train_test_split(data,
labels, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(trainX)
tfidf_test = tfidf_vectorizer.transform(testX)

print("\n training...\n")

clf = PassiveAggressiveClassifier(C=0.1, max_iter=1000)
#clf = SGDClassifier(max_iter=100, tol=1e-3)
#clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(tfidf_train, trainY)
preds = clf.predict(tfidf_test)

print("\n results...\n")

score = accuracy_score(testY, preds)
cm = confusion_matrix(testY, preds, labels=['FAKE','REAL'])
cl = classification_report(testY, preds)
print(f'Accuracy: {round(score*100,2)}%')
print(cm)
print(cl)



