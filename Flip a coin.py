# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:47:10 2023

@author: Rotoxe
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump

# Generar datos de entrenamiento y prueba
np.random.seed(0)
X_train = np.random.randint(2, size=(1000,1))
y_train = np.random.randint(2, size=(1000,1))
X_test = np.random.randint(2, size=(100,1))
y_test = np.random.randint(2, size=(100,1))

# Crear el modelo de regresión logística
log_reg = LogisticRegression()

# Entrenar el modelo con los datos de entrenamiento
log_reg.fit(X_train, y_train)

# Hacer predicciones en el conjunto de datos de prueba
y_pred = log_reg.predict(X_test)

# Calcular la precisión del modelo
accuracy = np.mean(y_pred == y_test)

print("La precisión del modelo es: ", accuracy)

dump(log_reg, 'L:', compress=0, protocol=None, cache_size=None)