# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:25:27 2022

@author: Juan Pablo Manzano
"""

# Repaso de conceptos 

import numpy as np
v = np.array([2,4,6,8,7])
m = np.array([[16,0,3,1,0],[6,7,0,4,0]])
print(m)
print(m[:,1])  # mostrar columna 1
print(m[1])  # mostrar fila 1 
print(m[1,:])  # tambien sirve para mostrar fila 1 
print(m[0,1])  # mostrar elemento 0,1
print(m[0,1:3])  # mostrar de la fila 0, el elemento 1 al 3 sin inculir 3


pares = v % 2 == 0  # aca se plantea la condicion de seleccion 
v_pares = v[pares]  # aca se pondria la condicion para la seleccion, puede tomarse de una variable o escribirse aca
print(v_pares)

v_may_5 = v[v > 5]  # aca se escribio la condicion en una sola linea, que muestra todos los valoers mayores a 5
print(v_may_5)

# Tambien es posible modificar los elementos con estas condiciones 

m[m==0] = 88  # aca cambiamos todos los valores == 0 en la matriz "m" por el valor 88

print(m)

# -------------------------- Practico 2 --------------------------------------

# Matrices y arrays

# ------------------------- Ejerciocio 1 -------------------------------------

A3 = np.array([[2,5,0],[7,3,8],[3,0,1]])
B2 = np.array([[2,5],[7,3],[3,0],[6,1],[-1,0]])

print(A3.ndim)
print(B2.ndim)

# ------------------------- Ejerciocio 2 -------------------------------------

# Hacer en papel

# ------------------------- Ejerciocio 3 -------------------------------------

# Hacer en papel

# ------------------------- Ejerciocio 4 -------------------------------------

alt = np.array([181.5,72.0,34.7,171.3,160.1])

print("el tipo de elemento utilizado para almacenar los datos es: " , type(alt))
print("el tipo de datos almacenados es: " ,alt.dtype)
print("el total de los familiares cargados es: " ,alt.size)

# ------------------------- Ejerciocio 5 -------------------------------------
lista = []
for i in range(8):
    lista.append(int(input("ingrese un numero entero")))

L = np.array(lista)

print(L)

# ------------------------- Ejerciocio 6 -------------------------------------

A1 = np.random.randint(0,6,size = 10)
print(A1)
A1[A1==0] = 6
print(A1)


# ------------------------- Ejerciocio 7 -------------------------------------

v_artesanal = [8,4,3,9,4,8,0,3,2,5]
v_art_inv = v_artesanal[-1::-1]

print(v_art_inv)

v_np = np.array(v_artesanal)

print(v_np[-1::-1])

# ------------------------- Ejerciocio 8 -------------------------------------

M1 = np.random.randint(1,10,size = (6,6))

print(M1)
I = np.identity(6)

print("Diagonal principal" , M1[I != 0])

# para hacerlo sin la matriz identidad 

lista = []

for i in range(len(M1[0])):
    for j in range(len(M1[0])):
        if i == j :
            lista.append(M1[i,j])

print(lista)
# ------------------------- Ejerciocio 9 -------------------------------------

v2 = np.linspace(1, 6, num = 50)

print(v2)

# ------------------------- Ejerciocio 10 ------------------------------------

M3 = np.random.randint(100,size = (200,100))

print(M3)

print(M3.dtype)  # con numeros enteros, no se si esa era la idea jajaja

# ------------------------- Ejerciocio 11 ------------------------------------

M4 = np.random.randint(100,size = (20,8))

print(M4)

print(np.sum(M4))   # para sumar todos los elementos se puede usar np.sum

# ------------------------- Ejerciocio 12 ------------------------------------

m_casera = []
for i in range(5):
    if i % 2 == 0:
        m = [1,1,1,1,1]
    else:
        m = [0,0,0,0,0]
    m_casera.append(m)
    

for i in range(len(m_casera)):
    print(m_casera[i], "\n")
    
M5 = np.full((5,5),0)

for i in range(5):
    if i % 2 == 0:  
        M5[i,:] = 1
    else:
        M5[i,:] = 0  # este paso no seria necesario si yo defini la primer martriz de 5x5 como todos "0"
        
print(M5)

# ------------------------- Ejerciocio 13 ------------------------------------

M6 = np.full((6,6),0)

for i in range(len(M6[0])):
    for j in range(len(M6[0])):
        if i == j:
            M6[i,j] = 1 
print(M6)

I = np.identity(6)

print(I)

# ------------------------- Ejerciocio 14 ------------------------------------

M7 = np.random.randint(0,2,size = (20,40),dtype = bool) # asi se puede crear una matriz booleana


M8 = np.where(M7,"Ocupado","Desocupado")  # con la funcion where se plantea una condicion, que en caso de cumplirse
                                          # reemplaza el valor con la primer opcion, y si no se cumple lo reemplaza con la segunda
                                          # en este caso no es una condicion, sino que la matriz M7 ya es de booleanos. 

print(M8)

# ------------------------- Ejerciocio 15 ------------------------------------

M9 = np.random.randint(100,size = (20,30))

es_par = np.where(M9 % 2 == 0,True,False)

print(es_par)

# ------------------------- Ejerciocio 16 ------------------------------------

# Funcion casera para sumar las matrices

def sum_mat(A,B,m,n):
    C = []
    c = []
    for i in range(m-1):
        for j in range(n-1):
            c.append(A[i,j]+B[i,j])
        C.append(c)
    return(C)

m = 4
n = 7
P1 = np.random.randint(1,11,size = (m,n))
P2 = np.random.randint(2,6,size = (m,n))

print(sum_mat(P1, P2, m, n))

# suma con Numpy

print(np.add(P1,P2))

# ------------------------- Ejerciocio 17 ------------------------------------

def mult_esc_mat (r,M):
    return(M*r)

P1 = np.random.randint(1,11,size = (m,n))
r = 5

print(P1)
print(mult_esc_mat(r, P1))

    
# ------------------------- Ejerciocio 18 ------------------------------------

# Se plantea una funcion, que detecte si el tipo de dato de entrada es una lista
# o no. En caso de ser una lista, multiplicara componente por componente. Y si es
# un array de numpy hara lo mismo pero con los metodos preestablecidos


def prod_punto (v1,v2):
    vr = []
    if not isinstance(v1, list):
        return(v1*v2)
    else:
        for i in range(len(v1)):
            vr.append(v1[i]*v2[i])
        return(vr)

v1 = np.array([1,2,8,9,3])
v2 = np.array([5,9,3,7,8])


print(prod_punto(v1, v2))

l1 = [1,2,8,9,3]
l2 = [5,9,3,7,8]

print(prod_punto(l1, l2))

# ------------------------- Ejerciocio 19 ------------------------------------

def prod_mat(A,B):
    if A.shape[1] == B.shape[0]:
        return(A.dot(B))
    else:
        return("las matrices no se pueden multiplicar")
    
R1 = np.array([[1,2,3,1],[2,4,1,0]]) 
R2 = np.array([[1,2],[2,7],[3,1],[0,3]])
print(R1)
print(R2)
print(prod_mat(R1, R2))





