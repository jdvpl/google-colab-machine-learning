# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
random_state=1)

# .target esta relacionado si tiene o no cancer

print(X_train.shape)
print(X_test.shape)
print(cancer.data)
print(cancer.target)