# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:35:32 2022

@author: jpman
"""

import matplotlib.pyplot as plt

xs = [0,1,2,3,4,5,6,7,8]
ys1 = [2*x for x in xs]
ys2 = [x**2 + 1for x in xs]

plt.plot(xs,ys1,"*--",label = "f(x) = 2*x",color="#0000cc") # creacion linea 1
plt.plot(xs,ys2,"o-r",label = "g(x) = x^2+1")               # creacion linea 2
plt.legend(loc=4)                                           # muestra leyendas y a donde van a ir
plt.title("Gráfico de prueba")
plt.xlabel("variable x")
plt.ylabel("variable y")
plt.grid(True)


# Creacion de lista para x

import numpy as np
xs = []
step = 0.05
ini = 3
fin = 10

xs = list(np.arange(ini,fin,step)) # Hace una lista desde el inicio al fin con el paso que queramos
xs2 = list(np.linspace(ini, fin,100)) # Hace una lista desde inicio a fin, con la cantidad de intervalos que queramos


