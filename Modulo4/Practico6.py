# -*- coding: utf-8 -*-
"""
Created on 10/06/2023

@author: Juan Pablo Manzano
"""
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Practico 6

# =============================================================================
# Ejercicio 1 Escala de grises
# =============================================================================

# NOTA IMPORTANTE, Ejecutar cada linea con su correspondiente semilla, sino los resultados aleatorios cambiaran
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
np.random.seed(7)  # Semilla para que esta matriz "aleatoria" sea siempre igual
m_random = np.random.randint(0, 255, size=(10, 10))

plt.imshow(m_random, cmap=plt.get_cmap('gray'))
plt.show()


# Nueva matriz de "patron a reconocer" con "cant" de valores cambiados al azar.
def mod_m(matriz, cant, semilla=7):
    p_mat = matriz.copy()
    np.random.seed(semilla)
    for i in range(cant):
        x_ind = np.random.randint(0, 10)
        y_ind = np.random.randint(0, 10)
        p_mat[x_ind, y_ind] = np.random.randint(0, 255)
    return p_mat


patron_1 = mod_m(m_random, 10, 7)

plt.imshow(patron_1, cmap=plt.get_cmap('gray'))
plt.show()


# def get_weight(memoria, patron):
#     size = memoria.shape[0]
#     m_vect = memoria.flatten()
#     p_vect = patron.T.flatten() # es necesario hacer la transpuesta de la matriz del patron a reconocer
#     W = np.multiply(m_vect, p_vect)
#     W = np.divide(W, len(m_vect))
#     W_mat = np.reshape(W, (size, size))
#     np.fill_diagonal(W_mat,0)
#     return W_mat

def get_weight(memoria, patron):
    m_vect = memoria.flatten()
    p_vect = patron.flatten()
    W = np.outer(m_vect, p_vect)
    W = np.divide(W, len(m_vect))
    W_mat = np.reshape(W, (memoria.size, memoria.size))
    np.fill_diagonal(W_mat, 0)
    return W_mat


# PRUEBA CON MATRIZ PEQUEÑA
matriz_chica = np.array([[0, 255, 0],
                         [0, 255, 255],
                         [255, 0, 0]])
print(get_weight(matriz_chica, matriz_chica))

matriz_w = get_weight(m_random, m_random)
print(matriz_w)

print(np.sum(matriz_w[5]))

# =============================================================================
# Ejercicio 1 Blanco y negro
# =============================================================================

np.random.seed(7)
memoria = np.random.choice([0, 255], size=(10, 10))
plt.imshow(memoria, cmap=plt.get_cmap('gray'))
plt.show()


def mod_mat(matriz, cant, modo = "ran", semilla=7):
    """
    :param matriz: matriz base para modificar
    :param cant: cantidad de pixeles a modificar
    :param modo: 2 modos posibles, colocar valor aleatorio con "ran"
                 o invertir los valores de pixelees aleatorios con "inv"
    :param semilla: semilla preestablecida en 7 para controlar resultados
    :return: devuelve una matriz del mismo tamaño que la ingresada con "cant"
             de pixeles cambiados por su inverso o alteatorio.
    """
    if modo == "ran":
        p_mat = matriz.copy()
        np.random.seed(semilla)
        for i in range(cant):
            x_ind = np.random.randint(0, matriz.shape[0])
            y_ind = np.random.randint(0, matriz.shape[0])
            p_mat[x_ind, y_ind] = random.choice([0, 255])
        return p_mat
    if modo == "inv":
        p_mat = matriz.copy()
        for i in range(cant):
            x_ind = np.random.randint(0, matriz.shape[0])
            y_ind = np.random.randint(0, matriz.shape[0])
            if matriz[x_ind, y_ind] == 0:
                p_mat[x_ind, y_ind] = 255
            else:
                p_mat[x_ind, y_ind] = 0
        return p_mat
    else:
        print("No existe el modo seleccionado")


patron1 = mod_mat(memoria,10,"inv")
plt.imshow(patron1, cmap=plt.get_cmap('gray'))
plt.show()


# Para obtener los pesos, hice algo distinto en lugar a lo que se plantea en la teoria.
# Hice que los valores valgan 1 o -1 para la matriz de pesos.

# FUNCION VIEJA
# def get_weight(memoria, patron):
#     m_vect = np.divide(memoria.flatten(),255)  # Divido por 255 para que valgan 1
#     p_vect = np.divide(patron.flatten(),255)
#     W = np.outer(m_vect, p_vect)
#     # W = np.divide(W, len(m_vect))  # Esto lo quité, porque no veo el sentido de aplicarlo.
#     W_mat = np.reshape(W, (memoria.size, memoria.size))
#     np.fill_diagonal(W_mat, 0)
#     return np.where(W_mat == 0 , -1, W_mat)  # En lugar de normalizar dividendo por N lo hago -1 o 1

def get_weight(memoria):
    m_vect = np.divide(memoria.flatten(),255)  # Divido por 255 para que valgan 1
    W = np.outer(m_vect, m_vect)
    # W = np.divide(W, len(m_vect))  # Esto lo quité, porque no veo el sentido de aplicarlo.
    W_mat = np.reshape(W, (memoria.size, memoria.size))
    W_mat = np.where(W_mat == 0, -1, W_mat)  # En lugar de normalizar dividendo por N lo hago -1 o 1
    np.fill_diagonal(W_mat, 0)
    return  W_mat

def get_weight2(memoria, patron):
    m_vect = memoria.flatten()
    p_vect = patron.flatten()
    W = np.zeros(len(m_vect) * len(p_vect))
    c = 0
    for i in m_vect:
        for j in p_vect:
            W[c] = i*j
            c += 1
    W = np.divide(W,len(m_vect))
    W_mat = np.reshape(W,(memoria.size, memoria.size))
    np.fill_diagonal(W_mat,0)
    return W_mat




matriz_w = get_weight(memoria)
print(matriz_w)

np.sum(matriz_w[0])
#
# matriz_chica = np.array([[-1, 1, -1],
#                          [-1, 1, 1],
#                          [1, -1, -1]])
#
# matriz_chica2 = np.array([[1, 1, -1],
#                          [1, -1, 1],
#                          [1, 1, 1]])
#
#
# matriz_w_chica = get_weight2(matriz_chica,matriz_chica)
# print(matriz_w_chica)
# print(np.sum(matriz_w_chica[5]))
# vector_resultado = np.dot(matriz_chica2.flatten(),matriz_w_chica)
# vector_resultado

print(np.dot(patron1.flatten(),matriz_w))


def norm(matriz):
    m_vectorizada = matriz.flatten()
    m_norm = np.divide(m_vectorizada,255)
    return np.where(m_norm == 0 , -1 , m_norm)


def func_eval (pesos,patron,trehold = 20):
    resulado = np.dot(norm(patron),pesos)
    resulado = np.where(resulado >= trehold, 255, resulado)
    resulado = np.where(resulado < trehold, 0, resulado)
    return np.reshape(resulado,(patron.shape[0],patron.shape[0]))
    # return resulado


np.random.seed(7)
memoria = np.random.choice([0, 255], size=(10, 10))
plt.imshow(memoria, cmap=plt.get_cmap('gray'))
plt.show()

patron1 = mod_mat(memoria,10,"inv")
plt.imshow(patron1, cmap=plt.get_cmap('gray'))
plt.show()

matriz_w = get_weight(memoria)
print(matriz_w)

# a

print(func_eval(matriz_w,patron1))
print(memoria)

patron2 = mod_mat(memoria,3,"inv")
plt.imshow(patron1, cmap=plt.get_cmap('gray'))
plt.show()

print(func_eval(matriz_w,patron2))
print(memoria)

patron3 = mod_mat(memoria,50,"inv")
plt.imshow(patron1, cmap=plt.get_cmap('gray'))
plt.show()

print(func_eval(matriz_w,patron3))
print(memoria)

# b

np.random.seed(7)
memoria2 = np.random.choice([0, 255], size=(10, 10))
plt.imshow(memoria2, cmap=plt.get_cmap('gray'))
plt.show()

np.random.seed(25)
memoria3 = np.random.choice([0, 255], size=(10, 10))
plt.imshow(memoria3, cmap=plt.get_cmap('gray'))
plt.show()

matriz_w = get_weight(memoria2) + get_weight(memoria3)

# Patron1

resultado1 = func_eval(matriz_w,patron1)
print(resultado1)

resultado2 = func_eval(matriz_w,resultado1)
print(resultado2)

# Patron2

resultado1 = func_eval(matriz_w,patron2)
print(resultado1)

resultado2 = func_eval(matriz_w,resultado1)
print(resultado2)

# Patron3

resultado1 = func_eval(matriz_w,patron3)
print(resultado1)

resultado2 = func_eval(matriz_w,resultado1)
print(resultado2)

# Los 3 patrones devolvieron la misma memoria2, pero si pruebo ingresando la memoria3 como patron devuelve esa

resultado1 = func_eval(matriz_w,memoria3)
print(resultado1)

resultado2 = func_eval(matriz_w,resultado1)
print(resultado2)

# Si pruebo con un patron ligeramente diferente a la memoria3 devuelve la memoria3

patron4 = mod_mat(memoria3,10,"inv",10)

resultado1 = func_eval(matriz_w,patron4)
print(resultado1)

resultado2 = func_eval(matriz_w,resultado1)
print(resultado2)

# Ahora si invierto 50 valores de la memoria 3, da un error...

patron5 = mod_mat(memoria3,80,"inv",100)

resultado1 = func_eval(matriz_w,patron5)
print(resultado1)

resultado2 = func_eval(matriz_w,resultado1)
print(resultado2)

print(patron5)
print(norm(patron5))
print(np.dot(norm(patron5),matriz_w))


# =============================================================================
# Ejercicio 4
# =============================================================================

# Tangente hiperbólica

x = np.linspace(-5, 5, 100)
coeficientes_B = [0.3, 1, 3]

fig, ax = plt.subplots()

for B in coeficientes_B:
    y = np.tanh(B * x)
    ax.plot(x, y, label=f'B = {B}')

ax.set_xlabel('x')
ax.set_ylabel('tanh(B * x)')
ax.set_title('Función Tangente Hiperbólica con diferentes coeficientes B')
ax.legend()

plt.show()

# RELU

fig, ax = plt.subplots()

for i in range(len(x)):
    if x[i] > 0:
        y[i] = x[i]
    else:
        y[i] = 0

ax.plot(x, y)
ax.set_xlabel('x')
ax.set_ylabel('RELU')
ax.set_title('Función RELU')
plt.show()

# Funcion definida

fig, ax = plt.subplots()

for B in coeficientes_B:
    y = ((np.e) ** (B * x)) / (1 + ((np.e) ** (B * x)))
    ax.plot(x, y, label=f'B = {B}')

ax.set_xlabel('x')
ax.set_ylabel('función definida')
ax.set_title('Función definida con diferentes coeficientes B')
ax.legend()

plt.show()

# =============================================================================
# Ejercicio 5
# =============================================================================
