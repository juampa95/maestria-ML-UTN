# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:55:19 2022

@author: Juan Pablo Manzano

"""
import math

# -----------------  Ejercicio 2 y 3  --------------------------

def dist(x1,x2,y1,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def recta(x1,x2,y1,y2):
    if x1 != x2:
        if y1 != y2:
            m = (x2-x1)/(y2-y1)
            b = y1 - m*x1 
        else:
            m = 0
            b = y1 - m*x1 
    else:
        m = "la linea es vertical"
        b = "no posee ordenada al origen"
    return (m,b)

# 2.a

x1 = 1
x2 = 5
y1 = 7
y2 = 2
print(" EJERCICOS 2 Y 3 a:")
print("Distancia: ", dist(x1,x2,y1,y2))
print("pendiente, ordenada ", recta(x1, x2, y1, y2))

# 2.b

x1 = 1
x2 = 2
y1 = 7
y2 = 7

print(" EJERCICOS 2 Y 3 b:")
print("Distancia: ",dist(x1,x2,y1,y2))
print("pendiente, ordenada ", recta(x1, x2, y1, y2))



# 2.c

x1 = 1
x2 = 1
y1 = 7
y2 = 2
print(" EJERCICOS 2 Y 3 c:")
print("Distancia: ",dist(x1,x2,y1,y2))
print("pendiente, ordenada ", recta(x1, x2, y1, y2))



# 2.d

x1 = -1
x2 = -5
y1 = 7
y2 = 2
print(" EJERCICOS 2 Y 3 d:")
print("Distancia: ",dist(x1,x2,y1,y2))
print("pendiente, ordenada ", recta(x1, x2, y1, y2))


# 2.e

x1 = -1
x2 = -3
y1 = -2
y2 = 2
print(" EJERCICOS 2 Y 3 e:")
print("Distancia: ",dist(x1,x2,y1,y2))
print("pendiente, ordenada ", recta(x1, x2, y1, y2))


# ----------------------  Ejercicio 4  -------------------------------
import numpy as np
ini = -10 
fin = 10
paso = 1

xs = list(np.arange(ini,fin,paso))
y1 = [2*x +1 for x in xs]
y2 = [6 - 3*x for x in xs]

# E4.a
a = 0
for x in xs :
    if 2*x+1 == 6-3*x :
        a = x
if a != 0:
    print("Las funciones se curzan en x = ", a)
else:
    print("Las funciones NO se cruzan")

# E4.b
a = 0
for x in xs:
    if 2*x+1 == 2*x - 3:
        a = x
if a != 0:
    print("Las funciones se curzan en x = ", a)
else:
    print("Las funciones NO se cruzan")
# E4.c
a = 0
for x in xs:
    if 1-2*x == 3*x+6:
        a = x
if a != 0:
    print("Las funciones se curzan en x = ", a)
else:
    print("Las funciones NO se cruzan")
# ----------------------  Ejercicio 5  ----------------------

import matplotlib.pyplot as plt

# 5.a

xs = list(np.arange(-2,2,0.1))
ys1 = [3*x**2 for x in xs]

plt.plot(xs, ys1)
plt.title("Ejercicio 5.a")

# 5.b

xs = list(np.arange(-2,2,0.1))
ys1 = [2*x**2 for x in xs]
ys2 = [3*x**2 for x in xs]
ys3 = [4*x**2 for x in xs]

plt.plot(xs, ys1,"--r",label = "f1(x)")
plt.plot(xs, ys2,":b",label = "f2(x)")
plt.plot(xs, ys3,"-g",label = "f3(x)")

plt.legend()
plt.title("Ejercicio 5.b")

# 5.c

xs = list(np.arange(-3,3,0.1))
ys1 = [math.cos(x) for x in xs]
ys2 = [math.sin(x) for x in xs]


plt.plot(xs, ys1,"--r",label = "coseno")
plt.plot(xs, ys2,":b",label = "seno")

plt.legend()
plt.title("Ejercicio 5.c")
          
# 5.d

xs = list(np.arange(-3,3,0.1))
xs2 =list(np.arange(0,3,0.1))
ys1 = [math.sqrt(x) for x in xs if x>0]
ys2 = [np.cbrt(x) for x in xs]


plt.plot(xs2, ys1,"--r",label = "raíz cuadrada")
plt.plot(xs, ys2,":b",label = "raíz cúbica")

plt.legend()
plt.title("Ejercicio 5.d")

# ----------------------  Ejercicio 6  ---------------------- En este si que me la quise complicar jajajaja 
import matplotlib.pyplot as plt
import numpy as np
import math
i = 0
dx = 0.0000000001
# xs = list(np.arange(2.8,5.5,0.1))
# xs2 =[]
# for i in range(len(xs)-1):
#     xs2.append(xs[i]+dx)

# ys1 = [5*x**5 -x**4 + 3*x**3 + x**2 -2*x - 8 for x in xs]
# ys2 = [5*x**5 -x**4 + 3*x**3 + x**2 -2*x - 8 for x in xs2]


# plt.plot(xs, ys1,"--",label = "f(x)")
# plt.plot(xs, ys1,":b",label = "f(x+dx)")

xs = [3.0,4.0,5.0,5.1,5.2,5.3,5.4]
x2 =[]
for i in range(len(xs)):
    x2.append(xs[i]+dx)

ys = [5*x**5 -x**4 + 3*x**3 + x**2 -2*x - 8 for x in xs]

pendientes = []
for i in range(len(xs)):  
    m = ((5*x2[i]**5 -x2[i]**4 + 3*x2[i]**3 + x2[i]**2 -2*x2[i] - 8) - (5*xs[i]**5 -xs[i]**4 + 3*xs[i]**3 + xs[i]**2 -2*xs[i] - 8))/dx
    pendientes.append(m)
    
ordenadas = []
for i in range(len(xs)):
    b = (5*xs[i]**5 -xs[i]**4 + 3*xs[i]**3 + xs[i]**2 -2*xs[i] - 8) - pendientes[i]*xs[i]
    ordenadas.append(b)
# print(ordenadas)
plt.plot(xs, ys,"or")

for i in range(len(xs)):
    dif = 0.1
    xtemp = [xs[i]-dif,xs[i]+dif]
    ytemp = [pendientes[i]*x + ordenadas[i] for x in xtemp]
    plt.plot(xtemp, ytemp)
    
    # Lo que quise hacer, fue grafica una recta, con la pendiente correspondiente a la pendiente en el punto
    # pero para ello fue necesario encontrar la ordenada al orgien de cada recta, por eso hice una lista
    # de ordenadas y una lista de pendientes. 
# ----------------------  Ejercicio 7  ----------------------

xs = list(np.arange(-4,4,0.1))
ys = [math.erf(x) for x in xs]

plt.plot(xs, ys)

# ----------------------  Ejercicio 8  ----------------------

xs = list(np.arange(-4,4,0.1))
ys=[]
for i in range(len(xs)):
    if xs[i]<0:
        ys.append(-1)
    if xs[i]==0:
        ys.append(0)
    if xs[i]>0:
        ys.append(1)

plt.plot(xs, ys)
            
# ----------------------  Ejercicio 9  ----------------------

xs = list(np.arange(-5,5,0.1))
ys=[]
for i in range(len(xs)):
    if xs[i]<0:
        ys.append(0)
    if xs[i]>=0:
        ys.append(1)

plt.plot(xs, ys)

# ----------------------  Ejercicio 10  ----------------------

xs = list(np.arange(-5,10,0.1))
ys = [math.tanh(x) for x in xs]

plt.plot(xs, ys)

# ----------------------  Ejercicio 11  ----------------------

# E11.a

xs = list(np.arange(-8,8,0.1))
ys = [x**2 for x in xs]

plt.plot(xs, ys)

# E11.b

xs = list(np.arange(-0.5,0.5,0.1)) # cambiar arange por (-3,3,0.1)

ys1 = [(math.e)**x for x in xs]
ys2 = [10**x for x in xs]
ys3 = [(1.7)**x for x in xs]

plt.plot(xs, ys1)
plt.plot(xs, ys2)
plt.plot(xs, ys3)

# E11.c

xs = list(np.arange(-2,2,0.1))
ys = [(1/x) for x in xs]

plt.plot(xs, ys)

# ----------------------  Ejercicio 12  ----------------------
import random
import os

print(os.getcwd())
os.chdir('C:/Users/jpman\Google Drive/Curso programacion/Python UTN/Machine Lerning/Modulo1')

arch = open("datos.txt","w",encoding="utf-8")
with arch:
    for i in range(1000):
        arch.write(str(random.uniform(-2.00,2.98)) + "," + str(random.uniform(0,0.99)) + "\n")

arch = open("datos.txt","r",encoding="utf-8")
lista=[]
with arch:
    datos = arch.readlines()
    for i in range(len(datos)):
        lista.append((datos[i].rstrip("\n")).split(",")) # VER COMANDO SPLIT MUY UTIL PARA LISTAS DE NUMEROS SEPARADOS POR COMA

xs = []
ys = []

for i in range(len(lista)):
    xs.append(float(lista[i][0]))
    ys.append(float(lista[i][1]))
    
plt.xlim(0, 2)
    
plt.plot(xs, ys,"o")

# ----------------------  Ejercicio 13  ----------------------

print("ingrese las 5 notas siendo \n"
      "1- Avances hogareños \n"
      "2- Entrega proyecto 1 \n"
      "3- Avances hogareños 2 \n"
      "4- Entrega proyecto 2 \n"
      "5- Examen Global \n")
notas = []
for i in range(5):
    while True:
        n = int(input("ingrese la nota correspondiente al N° " + str(i+1)))
        if(n>=0 and n<=10):
            break
        else:
            print("Recuerde que la nota debe estar entre 0 y 10")
    notas.append(n)
    
ponderacion = [0.1,0.2,0.1,0.2,0.4]
suma = 0
for i in range(len(notas)):
    suma = suma + notas[i]*ponderacion[i]
    
print("La nota final ponderada es", suma)
    

# ----------------------  Ejercicio 14  ----------------------

n = notas            # use los valores anteriores
p = ponderacion

def prom_ponderado (nota,peso):
    suma = 0
    for i in range(len(nota)):
        suma = suma + nota[i]*peso[i]
    return suma

print(prom_ponderado(n, p))