from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Leave one out función
def leave(X, y, alg):
	loo = LeaveOneOut()
	prediction = 0

	for train_index, test_index in loo.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		sm = SMOTE(random_state=12, k_neighbors=4)
		x_train_res, y_train_res = sm.fit_sample(X_train, y_train)
		alg.fit(x_train_res, y_train_res)
		pred = alg.predict(X_test)
		print(pred[0], y_test)
		if pred[0] == y_test:
			prediction = prediction + 1

	prediction = prediction/loo.get_n_splits(Xdata)
	print("acierto test", prediction)

# Cargar el dataset y selecionar X y Y
dataset = pd.read_csv('Fitoplancton_features.csv', sep=",", header=None)

X = dataset.drop(dataset.shape[1]-1, axis = 1) # remove output variable from input features
y = dataset[dataset.shape[1]-1]                # get only the output variable
print('X =',X.shape)
print('y =',y.shape)

Xdata = X.values # get values of features
Ydata = y.values # get output values
"""
##StandardScaler
scaler = StandardScaler()
scaler.fit(Xdata)
Xdata = scaler.transform(Xdata)
"""
# Medidas de los distintos modelos probados
alg = LinearDiscriminantAnalysis()
leave(Xdata, Ydata, alg)

# Guardar el modelo
clf = LinearDiscriminantAnalysis()
sm = SMOTE(random_state=12, k_neighbors=4)
x_train_res, y_train_res = sm.fit_sample(Xdata, Ydata)
clf.fit(x_train_res, y_train_res)

with open('modelo_fitoplancton.pkl', 'wb') as fid:
    pickle.dump(clf, fid) 