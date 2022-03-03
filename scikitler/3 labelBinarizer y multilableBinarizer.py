# -*- coding: utf-8 -*-
"""
asignacion de array a binarios
"""
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

#crear un feature
feature=np.array([["Jiren"],
                  ["Saitama"],
                  ["Tanjiro"],
                  ["Genos"],
                  ["Saitama"],
                  ["Jiren"]])

#llamar la clase lableBinarizer para tyransformar los datos

transformacion=LabelBinarizer()
#transforma los el te4xto a los numeros tener presente los que se repiten
array_transformado=transformacion.fit_transform(feature)

print(array_transformado)

#multiclase

feature_multiple=[ ("Peon", "Reina"),
                    ("Rey", "Alfil"),
                    ("Caballo", "Torre"),
                    ("Peon", "Torre"),
                    ("Reina", "Rey"),
                    ("Caballo", "Alfil"),
                  ]

#instanciar clase multilable

multiple_array=MultiLabelBinarizer()

data_multiple=multiple_array.fit_transform(feature_multiple)

print(data_multiple)