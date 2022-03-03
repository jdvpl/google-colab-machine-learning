# -*- coding: utf-8 -*-
"""
PCA para reducir dimensionalidad

KNN, PCA
"""
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

#knn
from sklearn.neighbors import KNeighborsClassifier

#instancias
knn = KNeighborsClassifier(n_neighbors=1)
cancer = load_breast_cancer()
scaler = StandardScaler()

# get  trains an test sets
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
random_state=1)


scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

# Numero de dimensiones
pca = PCA(n_components=3)
# .fit para transformar los datos
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print("Dimensiones originales: {}".format(str(X_scaled.shape)))
print("Dimensiones reducidas {}".format(str(X_pca.shape)))

#tener presente que se hace el calculo con los datos de trains
pca = PCA(n_components=10, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("X_train_pca.shape: {}".format(X_train_pca.shape))

#calcular distancia entre datos dee las dimensiones
knn.fit(X_train_pca, y_train)
print("Test set accuracy: {:.2f}".format(knn.score(X_test_pca, y_test)))
