# -*- coding: utf-8 -*-
"""
Created on 10/06/2023

@author: Juan Pablo Manzano
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Practico 6

# =============================================================================
# Ejercicio 1
# =============================================================================

m_random = np.random.randint(2,size=(10,10))

# genere una matriz aleatoria, y la copie aca para que quede fija, sino cada vez que se ejecute
# la linea de arriba, me crea una matriz diferente

m_almacenada = [[0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                [1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
                [1, 0, 1, 1, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 0, 1, 0, 0, 0, 0]]

m_almacenada = np.array(m_almacenada)
plt.imshow(m_almacenada,cmap=plt.get_cmap('gray'))
plt.show()

w_patron = np.empty((10,10))
for i in range(m_almacenada.shape[0]):
    for j in range(m_almacenada.shape[1]):
        w_patron[i,j] = (m_almacenada[i,j]*m_almacenada[j,i])/m_almacenada.size

w_patron2 = (1/m_almacenada.shape[0]) * np.dot(m_almacenada,m_almacenada.T)

w_patron2
m_vect = m_almacenada.flatten()





