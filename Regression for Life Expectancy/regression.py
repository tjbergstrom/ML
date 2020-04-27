# python3 regression.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from IPython.display import display
from sklearn.linear_model import LinearRegression
import statistics
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV


exp = pd.read_csv('Life Expectancy Data.csv')
exp = exp.dropna()


print("\n\nwhich variables actually affect life expectancy:\n")
data = exp.sample(frac=1.0)
filt = data.values[:,:]
i = 0
filt = filt[:,4:]
i = 4
r_sq = []
for col in filt.T:
    if i is not 3:
        col = col.astype(float)
        col = np.array(col)
        y =  data.values[:,3]
        y = y.astype(float)
        reg = LinearRegression().fit(col.reshape(-1,1), y)
        score = reg.score(col.reshape(-1,1), y)
        r_sq.append([data.columns[i],round(score,3)])
    i+=1
print(pd.DataFrame(r_sq,columns=["feature","r^2"]))


def print_coef(reg, X):
    data = []
    for coef, axis in zip(reg.coef_, X.axes[1]):
        data.append([axis,round(coef,3)])
    frame = pd.DataFrame(data,columns=["feature","coefficient"])
    display(frame)
    print("\nintercept:", round(reg.intercept_, 2))


def print_pred(reg, X_test, y_test, X_train, y_train):
    print("\npredictions:\n")
    y_pred = reg.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df.round(1))
    stdev = statistics.stdev(y_pred - y_test)
    print("\nstdev:", round(stdev,3))
    print("score:", round(reg.score(X_train, y_train),3))


exp = pd.read_csv('Life Expectancy Data.csv')
exp = exp.dropna()
exp.replace(to_replace = "Developed", value = 1.0, inplace=True)
exp.replace(to_replace = "Developing", value = 0.0, inplace=True)
X = exp.drop('Life expectancy ', axis=1)
X = X.drop('Country', axis=1)
X = X.drop('Year', axis=1)
Y = exp['Life expectancy ']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
x_orig = X_train
y_orig = y_train
x_t_orig = X_test
y_t_orig = y_test
reg = LinearRegression()
reg.fit(X_train, y_train)
print("\n\ncoefficients after training:\n")
print_coef(reg, X)
print_pred(reg, X_test, y_test, X_train, y_train)


X = X.drop(' thinness 5-9 years', axis=1)
X = X.drop('Population', axis=1)
X = X.drop('GDP', axis=1)
X = X.drop('Polio', axis=1)
X = X.drop('Measles ', axis=1)
X = X.drop('percentage expenditure', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
reg.fit(X_train, y_train)
print("\n\ncoefficients after dropping variables:\n")
print_coef(reg, X)
print_pred(reg, X_test, y_test, X_train, y_train)


X = X.drop(' thinness  1-19 years', axis=1)
X = X.drop(' HIV/AIDS', axis=1)
X = X.drop('Diphtheria ', axis=1)
X = X.drop('Total expenditure', axis=1)
X = X.drop('Hepatitis B', axis=1)
X = X.drop('Alcohol', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
reg.fit(X_train, y_train)
print("\n\ncoefficients after dropping more variables:\n")
print_coef(reg, X)
print_pred(reg, X_test, y_test, X_train, y_train)


X_train = x_orig
y_train = y_orig
X_test = x_t_orig
y_test = y_t_orig
X = exp.drop('Life expectancy ', axis=1)


print("\n\n")
reg = Ridge(alpha = 0.2, max_iter = 10000)
reg.fit(X_train, y_train)
print("\n\ncoefficients after Ridge():\n")
print_coef(reg, X)
print_pred(reg, X_test, y_test, X_train, y_train)


reg = Lasso(alpha = 0.2, max_iter = 10000)
reg.fit(X_train, y_train)
print("\n\ncoefficients after Lasso():\n")
print_coef(reg, X)
print_pred(reg, X_test, y_test, X_train, y_train)


print("\ncross validation:\n")
lasso_cv = LassoCV(alphas = [.01, .5, .8], random_state = 0, max_iter = 10000)
k_fold = KFold(4)
for k, (train, test) in enumerate(k_fold.split(X_train, y_train)):
    lasso_cv.fit(X_train, y_train)
    if k is 1:
        lasso_cv.alpha_ = 0.2
    if k is 2:
        lasso_cv.alpha_ = 0.5
    if k is 3:
        lasso_cv.alpha_ = 0.8
    print("fold =", k, ", alpha =", lasso_cv.alpha_, ", score =", round(lasso_cv.score(X_train, y_train),4))


train = exp.sample(frac=0.8)
life_x = train.values[:,3]
regr = LinearRegression().fit(train.values[:,4:], life_x)
test = exp.sample(frac=0.2)
print("\n\n\nhow well does country predict:\n")
print("prediction - actual")
cnt = 0
country_arr = {}
tot = 0
ea = 0
for row in test.values:
    prediction = regr.predict(row[4:].reshape(1, -1))[0]
    actual = row[3]
    if cnt<10:
        print("    ", round(prediction,1), "    ", actual, "  -", row[0])
    cnt += 1
    try:
        country_arr[row[0]].append(prediction - actual)
    except:
        country_arr[row[0]] = [prediction - actual]
country_stdv = []
for i in country_arr:
    stv = 0
    if(len(country_arr[i]) > 1):
        stdv = round(statistics.stdev(country_arr[i]), 3)
    country_stdv.append([i, stdv])
    if stdv != 0.0:
        ea += stdv
        tot += 1
df = pd.DataFrame(country_stdv, columns = ["country", "stdev"])
print("\n\nstdev for each country:\n")
print(df)
print("\nstdev avg for all countries:", round(ea/tot, 3))


print("\n\n\ntask failed successfully\n\n\n")



