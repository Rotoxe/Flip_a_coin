# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:47:10 2023

@author: Rotoxe
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump
from joblib import load


#cargar modelo
log_reg = load('modelo.joblib')

# Generar datos de entrenamiento y prueba
np.random.seed(0)
# X son las etiquetas
X_train = np.array([[8.06],[8.00],[7.59],[7.58],[7.56],[7.55],[7.54],[7.53],[7.42],[7.41],[7.40],[7.39],[7.38],[7.31],[7.30],[7.29],[7.26],[7.24],[7.23],[7.21],[7.20],[7.11],[7.10],[7.09],[7.08],[7.07],[7.06],[7.05],[7.04],[7.02],[7.01],[7.00],[6.59],[6.58],[6.57],[6.56],[6.55],[6.53],[6.52],[6.25],[6.28],[6.29],[6.35],[6.36],[6.39],[6.41],[6.42],[6.43],[6.44],[6.45],[6.48],[6.49],[6.50],[6.51]])
# y son cara o cruz
y_train = np.array([[0],[2],[0],[0],[0],[0],[0],[0],[1],[0],[0],[1],[1],[0],[0],[1],[0],[1],[0],[1],[0],[0],[0],[1],[0],[0],[0],[0],[1],[0],[0],[1],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[0],[0],[0],[1],[1],[0],[0],[1],[0],[0],[0],[0]])

# estos son los datos que se utilizarán para probar 
X_test = np.array([[7.52],[7.51],[7.50],[7.49],[7.48],[7.47],[7.46],[7.45],[7.44],[7.43],[7.37],[7.36],[7.35],[7.34],[7.33],[7.32],[7.28],[7.27],[7.25],[7.22],[7.19],[7.18],[7.12],[6.31],[6.32],[6.33],[6.34],[6.38],[6.47],[6.54],[7.03],[7.13],[7.14],[7.15],[7.16],[7.17]])
y_test = np.array([[1],[0],[0],[0],[0],[0],[1],[0],[0],[1],[0],[1],[0],[0],[0],[1],[0],[1],[0],[0],[0],[0],[0],[0],[0],[0],[1],[0],[0],[0],[0],[0],[0],[0],[1],[1]])


# Crear el modelo de regresión logística
log_reg = LogisticRegression()

# Entrenar el modelo con los datos de entrenamiento
log_reg.fit(X_train, y_train)

# Hacer predicciones en el conjunto de datos de prueba
y_pred = log_reg.predict(X_test)

# Calcular la precisión del modelo
accuracy = np.mean(y_pred == y_test)


print("La precisión del modelo es: ", accuracy)

#crear una nueva muestra con una sola fila y el mismo número de columnas
nueva_muestra = np.array([[8.59],[8.58],[8.57],[8.56],[8.55],[8.54],[8.53],[8.52],[8.51],[8.50],[8.49],[8.48],[8.47],[8.46],[8.45],[8.44],[8.43],[8.42],[8.41],[8.40],[8.39],[8.38],[8.37],[8.36],[8.35],[8.34],[8.33],[8.32],[8.31],[8.30],[8.29],[8.28],[8.27],[8.26],[8.25],[8.24],[8.23]])

# Hacer una predicción con la nueva muestra
predicción = log_reg.predict(nueva_muestra)

print("la predicción es: ", predicción)

# Generar una lista de valores para los que quieres hacer predicciones
valores = np.array([[8.59],[8.58],[8.57],[8.56],[8.55],[8.54],[8.53],[8.52],[8.51],[8.50],[8.49],[8.48],[8.47],[8.46],[8.45],[8.44],[8.43],[8.42],[8.41],[8.40],[8.39],[8.38],[8.37],[8.36],[8.35],[8.34],[8.33],[8.32],[8.31],[8.30],[8.29],[8.28],[8.27],[8.26],[8.25],[8.24],[8.23]])

# Hacer predicciones con la lista de valores
predicciones = log_reg.predict(valores)

# Crear una lista con los índices en los que se hizo una predicción de 1
indices_1 = [i for i, x in enumerate(predicciones) if x == 1]

# Imprimir la lista de índices en los que se hizo una predicción de 1
print("En los siguientes índices se hizo una predicción de 1:", indices_1)

dump(log_reg, 'modelo.joblib', compress=0, protocol=None, cache_size=None)
