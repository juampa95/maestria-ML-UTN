# -*- coding: utf-8 -*-
"""
Created on 10/04/2023

@author: Juan Pablo Manzano
"""
import numpy as np
import pandas as pd
import os

# Practico 5

# =============================================================================
# Ejercicio 1
# =============================================================================
# modo directo

x = [2, 3, 4, 5, 6, 7]
y = [18, 33, 37, 54, 59, 71]

x_prom = np.mean(x)
y_prom = np.mean(y)
m = 0
difx = 0
for j in range(len(x)):
    difx = difx + (x[j] - x_prom) ** 2
for i in range(len(x)):
    m = m + ((x[i] - x_prom) * (y[i] - y_prom)) / (difx)

b = y_prom - m * x_prom

print(f'la recta de ajuste es {round(m, 2)}x + ({round(b, 2)})')

y_pred = []
for i in range(len(x)):
    yp = m * x[i] + b
    y_pred.append(yp)

y_pred
sum = 0
for i in range(len(x)):
    sum = sum + ((y[i] - y_pred[i]) ** 2)
j = sum / (2 * len(x))

j

print(f'la funcion costo me da {round(j, 2)}. Es distinto al valor 18,95 del practico \n'
      f'creo que es porque en el PDF se olvidaron de dividirlo por 2N')

# Creamos una funcion que me determine 'm','b'y el valor de la funcion de costo
def reg_lin_una_var(xs, ys):
    x_prom = np.mean(xs)
    y_prom = np.mean(ys)
    m = 0
    difx = 0
    for j in range(len(xs)):
        difx = difx + (xs[j] - x_prom) ** 2
    for i in range(len(xs)):
        m = m + ((xs[i] - x_prom) * (ys[i] - y_prom)) / (difx)
    b = y_prom - m * x_prom
    sum = 0
    for i in range(len(xs)):
        sum = sum + ((ys[i] - (m * xs[i] + b)) ** 2)
    j = sum / (2 * len(xs))
    return [m, b, j]

resultado = reg_lin_una_var(x,y)
print(f'la recta de ajuste tiene un m = {resultado[0]}\n'
      f'la recta de ajuste tiene un b = {resultado[1]}\n'
      f'la funcion de costo tiene un valor de j = {resultado[2]}')



# =============================================================================
# Ejercicio 2
# =============================================================================
print(os.getcwd())
os.chdir('D:\gitProyects\maestria-ML-UTN\Modulo3\Practico5')

datos = pd.read_csv('heights_weights.csv')

datos

x_2 = datos['Height'].tolist()
y_2 = datos['Weight'].tolist()

resultado_2 = reg_lin_una_var(x_2,y_2)

print(f'la recta de ajuste tiene un m = {resultado_2[0]}\n'
      f'la recta de ajuste tiene un b = {resultado_2[1]}\n'
      f'la funcion de costo tiene un valor de j = {resultado_2[2]}')

print(f'estimación del peso de una mujer de 1.3m = {resultado_2[0]*1.3+resultado_2[1]}')
print(f'estimación del peso de una mujer de 1.6m = {resultado_2[0]*1.6+resultado_2[1]}')
print(f'estimación del peso de una mujer de 2.0m = {resultado_2[0]*2.0+resultado_2[1]}')

# =============================================================================
# Ejercicio 3
# =============================================================================

