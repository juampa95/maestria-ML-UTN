# -*- coding: utf-8 -*-
"""
Created on 10/04/2023

@author: Juan Pablo Manzano
"""
import numpy as np

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
    difx = difx + (x[j]-x_prom)**2
for i in range(len(x)):
    m = m + ((x[i] - x_prom)*(y[i] - y_prom))/(difx)

b = y_prom - m * x_prom

print(f'la recta de ajuste es {round(m,2)}x + ({round(b,2)})')

y_pred = []
for i in range(len(x)):
    yp = m*x[i]+b
    y_pred.append(yp)

y_pred
sum = 0
for i in range(len(x)):
    sum = sum + ((y[i]-y_pred[i])**2)
j = sum/(2*len(x))

j

print(f'la funcion costo me da {round(j,2)}. Es distinto al valor 18,95 del practico \n'
      f'creo que es porque en el PDF se olvidaron de dividirlo por 2N')