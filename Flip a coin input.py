# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:47:10 2023

@author: Rotoxe
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
#from joblib import dump
from joblib import load

#cargar modelo
log_reg = load('modelo.joblib')

# Lista de datos de entrada
X_inputs = []

# Lista de etiquetas
labels = ["sol", "aguila"]
# Asignar un número a la etiqueta
Labels_numeric = [0,1]

# Pedir los datos

ask_again = True
while ask_again:
    X_input = int(input("ingrese un número (0 o 1): "))
    X_input = np.array(X_input).reshape(-1,1)
    X_inputs.append(X_input)

    # Asignar la etiqueta correcta
    y_label = int(input("ingresae la etiqueta 0 o 1: "))
    y_label = np.array([Labels_numeric[y_label]])
    
    # Ingresar otro input
    response = input("¿Agregar otro input? (si/no): ")
    if response.lower() == "no":
        ask_again = False
    
# Concatenar todos los datos de entrada en una sola matriz
X_input = np.concatenate(X_inputs, y_label)

# Crear el modelo de regresión logística
Log_reg = LogisticRegression()
    
# Entrenar el modelo con los datos de entrenamiento
Log_reg.fit(np.concatenate(X_inputs), y_label.ravel())
    
#Hacer predicciones en el conjunto de datos de prueba
y_pred = log_reg.predict(X_input)

# Calcular la precisión del modelo
#accuracy = np.mean(y_pred == y_test)

# Imprimir resultados de la predicción
print("El resultado de la predicción es: ", y_pred)

#dump(log_reg, 'modelo.joblib', compress=0, protocol=None, cache_size=None)
