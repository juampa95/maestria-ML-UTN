# -*- coding: utf-8 -*-
"""
Created on 10/04/2023

@author: Juan Pablo Manzano
"""
import datetime

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import datetime, timedelta

os.chdir('D:\gitProyects\maestria-ML-UTN\Modulo3\Practico5')

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


resultado = reg_lin_una_var(x, y)
print(f'la recta de ajuste tiene un m = {resultado[0]}\n'
      f'la recta de ajuste tiene un b = {resultado[1]}\n'
      f'la funcion de costo tiene un valor de j = {resultado[2]}')

# =============================================================================
# Ejercicio 2
# =============================================================================
print(os.getcwd())

datos = pd.read_csv('heights_weights.csv')

datos

x_2 = datos['Height'].tolist()
y_2 = datos['Weight'].tolist()

resultado_2 = reg_lin_una_var(x_2, y_2)

print(f'la recta de ajuste tiene un m = {resultado_2[0]}\n'
      f'la recta de ajuste tiene un b = {resultado_2[1]}\n'
      f'la funcion de costo tiene un valor de j = {resultado_2[2]}')

print(f'estimación del peso de una mujer de 1.3m = {resultado_2[0] * 1.3 + resultado_2[1]}')
print(f'estimación del peso de una mujer de 1.6m = {resultado_2[0] * 1.6 + resultado_2[1]}')
print(f'estimación del peso de una mujer de 2.0m = {resultado_2[0] * 2.0 + resultado_2[1]}')

# =============================================================================
# Ejercicio 3
# =============================================================================

print(os.getcwd())

datos = pd.read_csv('notas_horasestudio.csv')

datos

x_3 = datos['Horas'].tolist()
y_3 = datos['Notas'].tolist()

resultado_3 = reg_lin_una_var(x_3, y_3)

resultado_3
print(f'la recta de ajuste tiene un m = {resultado_3[0]}\n'
      f'la recta de ajuste tiene un b = {resultado_3[1]}\n'
      f'la funcion de costo tiene un valor de j = {resultado_3[2]}')

# =============================================================================
# Ejercicio 4
# =============================================================================

x_4 = [4.1, 6, 6, 7.3, 7.8, 8.1, 8.77, 9.45, 12.3, 14.2, 15, 16, 18.12]
y_4 = [90, 89, 83, 81, 73, 71, 68, 60, 59, 54, 35, 35, 21]

# MODO DIRECTO CON FUNCION CREADA
resultado_4_directo = reg_lin_una_var(x_4, y_4)
print(f'la recta de ajuste tiene un m = {resultado_4_directo[0]}\n'
      f'la recta de ajuste tiene un b = {resultado_4_directo[1]}\n'
      f'la funcion de costo tiene un valor de j = {resultado_4_directo[2]}')

# MODO POR FUERZA BRUTA
i = 0
resultados_4_fbruta = pd.DataFrame(columns=['m', 'b', 'j'])
for m in range(-10, 0):
    for b in range(100, 200):
        sum = 0
        for x in range(len(x_4)):
            ypred = m * x_4[x] + b
            sum = sum + ((y_4[x] - (m * x_4[x] + b)) ** 2)
        j = sum / (2 * len(x_4))
        resultados_4_fbruta.loc[i] = {'m': m, 'b': b, 'j': j}
        i += 1

r4_fb = resultados_4_fbruta.sort_values('j', ascending=True).head(1)

print(f'la recta de mejor ajuste sera la que tenga los siguientes parametros\n'
      f'y su funcion de costo valdra lo que marca la tabla en la columna j\n'
      f'{r4_fb}')

# MODO DESCENSO POR GRADIENTE

# concepto de derivada, cuanto crece la función, cuando el incremento tiende a 0
# la función incrementada, menos la función sin incrementar sobre el incremento

def fcosto(xs,ys,m,b):
    sum = 0
    for i in range(len(xs)):
        sum = sum + ((ys[i] - (m * xs[i] + b)) ** 2)
    j = sum / (2 * len(xs))
    return j

def der_m(xs,ys,m,b):
    sum = 0
    dm = 0
    for i in range(len(xs)):
        sum = sum + (xs[i]*(ys[i]-m*xs[i]-b))
    dm = -2*sum/len(xs)
    return dm

def der_b(xs,ys,m,b):
    sum = 0
    db = 0
    for i in range(len(xs)):
        sum = sum +(ys[i]-m*xs[i]-b)
    db = -2*sum/len(xs)
    return db

m = -10
b = -10
Lm = 0.0001
Lb = 1
costo = 10000000000
r4_descGrad = pd.DataFrame(columns=['iter','m','b','j'])

for i in range(1000):
    new_costo = fcosto(x_4,y_4,m,b)
    if costo > new_costo:
        costo = new_costo
    r4_descGrad.loc[i] = {'iter':i,'m': m, 'b': b, 'j': new_costo}
    m = m - (Lm*der_m(x_4,y_4,m,b))
    b = b - (Lb*der_b(x_4,y_4,m,b))

r4_descGrad.sort_values('j',ascending=True)
costo

sns.lineplot(data = r4_descGrad,
             x= 'iter',
             y= 'j')
plt.show()
print(costo)
# =============================================================================
# Ejercicio 5
# =============================================================================

dolar = pd.read_csv('fecha_dolar_bna.csv')
dolar.info()

#Convertimos la columna indice_tiempo en un datetime
dolar['indice_tiempo'] = pd.to_datetime(dolar['indice_tiempo'],format='%d/%m/%Y')
dolar = dolar.reset_index()
# De esta forma, la columna index corresponde a un numero entero para cada fecha
dolar.head(20)

# viendo que los dias estan continuados (incluye sabados y domingos) podemos
# tomar como fecha inicial el primer valor y sumarle dias con datetime

star_time = dolar.sort_values('indice_tiempo',ascending=True).loc[0][1]
star_time

# Se podria buscar el numero correspondiente a cualquier dia de esta otra forma
dif_dias = (datetime.strptime('01/11/2023','%d/%m/%Y') - star_time).days
dif_dias

# Y podriamos buscar el dia correspondiente a cualquier numero de esta forma
num_a_dia = (star_time + timedelta(days=200)).strftime('%d/%m/%Y')
num_a_dia

x_5 = dolar['index'].tolist()
y_5 = dolar['tipo_cambio_bna_vendedor'].tolist()

m = 0
b = -20
Lm = 0.0000001
Lb = 0.01
costo = 10000000000
r5_descGrad = pd.DataFrame(columns=['iter','m','b','j'])

for i in range(1000):
    new_costo = fcosto(x_5,y_5,m,b)
    if costo > new_costo:
        costo = new_costo
    r5_descGrad.loc[i] = {'iter':i,'m': m, 'b': b, 'j': new_costo}
    m = m - (Lm*der_m(x_5,y_5,m,b))
    b = b - (Lb*der_b(x_5,y_5,m,b))

# Obtenemos los mejores valores de m y b y el costo
m_best = r5_descGrad.sort_values('j',ascending=True).head(1)['m'].values[0]
b_best = r5_descGrad.sort_values('j',ascending=True).head(1)['b'].values[0]
costo

# cree un df para graficar la linea de prediccion
prediccion = pd.DataFrame({'x':x_5,'y':[m_best*x + b_best for x in x_5]})

sns.lineplot(data = dolar,
                x = 'index',
                y = 'tipo_cambio_bna_vendedor')
sns.lineplot(data = prediccion,
             x = 'x',
             y = 'y',
             color = 'red')
plt.show()

# En el grafico podemos ver que la recta no se ajusta tan bien a los datos.

nov23 = datetime.strptime('01/11/2023','%d/%m/%Y')
mar24 = datetime.strptime('01/03/2024','%d/%m/%Y')
abr24 = datetime.strptime('01/04/2024','%d/%m/%Y')


print(f'el valor para 01/11/23 es {m_best*(nov23-star_time).days+b_best}\n'
      f'el valor para 01/03/24 es {m_best*(mar24-star_time).days+b_best}\n'
      f'el valor para 01/04/24 es {m_best*(abr24-star_time).days+b_best}\n'
      f'El resultado no es bueno, porque la recta no se aproxima a la realidad')

# =============================================================================
# Ejercicio 6
# =============================================================================

