# -*- coding: utf-8 -*-
"""
minmaxescaler

.fit para preprocesar los datos de entrada
"""

from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
random_state=1)

scaler = MinMaxScaler()

# informacion del minimo y maximo de los datos

scaler.fit(X_train)

# transformar los datos varian entre 0 y 1 los de entrenamiento

X_train_scaled = scaler.transform(X_train)


# transformar los datos varian entre 0 y 1 los de prueba

X_test_scaled = scaler.transform(X_test)

print("datos de training")
print(X_train_scaled)


print("datos de test")
print(X_test_scaled)