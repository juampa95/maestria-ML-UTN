# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 12:24:34 2022

@author: Juan Pablo Manzano
"""
#   Practico N°3 

# =============================================================================
# Ejercicio N°1
# =============================================================================

from PIL import Image
import numpy as np
import cv2

red = Image.new('RGB', (600,800),color = 'red')

green = Image.new('RGB', (600,800),color = 'green')

yellow = Image.new('RGB', (600,800),color = 'yellow')

red.save('rojo.jpg')
green.save('verde.jpg')
yellow.save('amarillo.jpg')


# =============================================================================
# Ejercicio N°2
# =============================================================================

cuadrado = Image.open('cuadrado.jpg')

alto,ancho = cuadrado.size

pixels = np.array(cuadrado)

for i in range(ancho):
    for j in range(alto):
        if pixels[i,j][0] != 255 or pixels[i,j][1] != 255 or pixels[i,j][2] != 255:
            pixels[i,j][1] = 255
            
cuadrado_verde = Image.fromarray(pixels)

cuadrado_verde.show()

# =============================================================================
# Ejercicio N°3 
# =============================================================================

arcoiris = Image.new('RGB',(600,300),color = 'white')

alto,ancho = arcoiris.size
p = np.array(arcoiris)

for i in range(alto):
    for j in range(ancho):
        if j < 50:
            p[j,i] = (255,100,0)
        if j >= 50:
            p[j,i] = (0,255,0)
        if j >= 100:            
            p[j,i] = (10,0,255)
        if j >= 150:
            p[j,i] = (100,0,210)
        if j >= 200:
            p[j,i] = (100,100,35)
        if j >= 250:
            p[j,i] = (30,200,100)
            
arco = Image.fromarray(p)

arco.show()

# =============================================================================
# Ejercicio N°4
# =============================================================================

def saturar (img,s):
    imagen = Image.open(img)
    alto,ancho = imagen.size
    matriz = np.array(imagen.convert('HSV'))
    for i in range(ancho):
        for j in range(alto):
            matriz[i,j][1] = s
    return(Image.fromarray(matriz).show())

saturar('cuadrado.jpg',1)


# =============================================================================
# Ejercicio N°5
# =============================================================================



matriz_base = np.full((600,800,3),125,dtype = 'uint8')

# cv2.imshow('matriz',matriz_base)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


alto,ancho,dim = matriz_base.shape
for i in range(alto):
    for j in range(ancho):
        matriz_base[i,j] = (0,250,0)    # El alto y ancho es al reves que pillow

cv2.imwrite('verde_cv.jpg',matriz_base)

for i in range(alto):
    for j in range(ancho):
        matriz_base[i,j] = (0,0,255)    

cv2.imwrite('rojo_cv.jpg',matriz_base)

for i in range(alto):
    for j in range(ancho):
        matriz_base[i,j] = (0,255,255)  

cv2.imwrite('amarillo_cv.jpg',matriz_base)

# =============================================================================
# Ejercicio N°6
# =============================================================================

p = np.full((500,300,3),125,dtype = 'uint8')
alto,ancho,dim = p.shape

for i in range(alto):
    for j in range(ancho):
        if j < 50:
            p[i,j] = (255,100,0)
        if j >= 50:
            p[i,j] = (0,255,0)
        if j >= 100:            
            p[i,j] = (10,0,255)
        if j >= 150:
            p[i,j] = (100,0,210)
        if j >= 200:
            p[i,j] = (100,100,35)
        if j >= 250:
            p[i,j] = (30,200,100)

cv2.imwrite('verticales.jpg',p)

# =============================================================================
# Ejercicio N°7
# =============================================================================

p = np.full((600,600,3),125,dtype = 'uint8')
alto,ancho,dim = p.shape

r = int(input("ingrese el valor RGB para canal ROJO"))
g = int(input("ingrese el valor RGB para canal VERDE"))
b = int(input("ingrese el valor RGB para canal AZUL"))


for i in range(alto):
    for j in range(ancho):
        if i > 250 and i < 350 and j > 250 and j < 350:
            p[i,j]= (0,0,0)
        else:
            p[i,j]= (b,g,r)

                
cv2.imshow('imagen',p)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
# Ejercicio N°8
# =============================================================================

img = cv2.imread('puros.png')

alto,ancho,dim = img.shape

# print(img[int(alto/2),int(ancho/2)]) # es rojo pleno (0,0,255)

for i in range(alto):
    for j in range(ancho):
        if img[i,j][0] == 0 and img[i,j][1] == 0 and img[i,j][2] == 255:
            img[i,j] = (0,255,255)

cv2.imshow('imagen',img)  # No se porque se ve una pequeña linea roja
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
# Ejercicio N°9
# =============================================================================

def padding (imagen,n):
    img = cv2.imread(imagen)
    alto,ancho,dim = img.shape
    p_nueva = np.full((alto+2*n,ancho+2*n,3),255,dtype = 'uint8')
    for i in range(alto+(2*n)):
        for j in range(ancho+(2*n)):
            if i >= n and i < alto+n:
                if j >= n  and j < ancho+n:
                    p_nueva[i,j] = img[i-n,j-n]
                    
    return(cv2.imwrite('imagen_padding.jpg',p_nueva))


padding('rojo_cv.jpg', 50)

# =============================================================================
# Ejercicio N°10
# =============================================================================

img = cv2.imread('puros.png')

alto,ancho,dim = img.shape

for i in range(alto):
    for j in range(ancho):
        img[i,j][0] = 255 - img[i,j][0]
        img[i,j][1] = 255 - img[i,j][1]
        img[i,j][2] = 255 - img[i,j][2]

cv2.imshow('imagen',img)  # No se porque se ve una pequeña linea roja
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
# Ejercicio N°11
# =============================================================================

# Segun entiendo, debemos hacer una funcion para quitar el canal del color rojo
# verde o azul. Pero esto no significa que solo vaya a afecatar al color rojo.
# Porque por ejemplo, el naranja tambien tiene x cantidad del canal rojo

def quitar_rojo (imagen):
    img = cv2.imread(imagen)
    alto,ancho,dim = img.shape
    for i in range(alto):
        for j in range(ancho):
            img[i,j][2] = 0
    return(cv2.imwrite('sin_rojo.jpg',img))

quitar_rojo('puros.png')

def quitar_verde (imagen):
    img = cv2.imread(imagen)
    alto,ancho,dim = img.shape
    for i in range(alto):
        for j in range(ancho):
            img[i,j][1] = 0
    return(cv2.imwrite('sin_verde.jpg',img))

def quitar_azul (imagen):
    img = cv2.imread(imagen)
    alto,ancho,dim = img.shape
    for i in range(alto):
        for j in range(ancho):
            img[i,j][0] = 0
    return(cv2.imwrite('sin_azul.jpg',img))

# =============================================================================
# Ejercicio N°12
# =============================================================================

def hamming_pal (a,b):
    cuenta = 0
    for i in range(len(a)):
        if  a[i] != b[i]:
            cuenta = cuenta + 1
    return(cuenta)
            
# =============================================================================
# Ejercicio N°13
# =============================================================================

def hamming (a,b):
    imga = cv2.imread(a)
    imgb = cv2.imread(b)
    alto,ancho,dim = imga.shape
    contador = 0
    for i in range(alto):
        for j in range(ancho):
            if imga[i,j][0] != imgb[i,j][0] and imga[i,j][1] != imgb[i,j][1] and imga[i,j][2] != imgb[i,j][2]:
                contador = contador + 1
    return(contador)

# =============================================================================
# Ejercicio N°14
# =============================================================================

p1 = np.random.randint(255,size=(600,600),dtype = np.uint8)
p2 = np.random.randint(255,size=(600,600),dtype = np.uint8)

cv2.imwrite('r1.jpg',p1)
cv2.imwrite('r2.jpg',p2)

hamming('r1.jpg', 'r2.jpg')

# =============================================================================
# Ejercicio N°15
# =============================================================================

img = cv2.imread('perro.jpg')

alto,ancho,dim = img.shape
red = np.full(img.shape,0,dtype = 'uint8')
green = np.full(img.shape,0,dtype = 'uint8')
blue = np.full(img.shape,0,dtype = 'uint8')

for i in range(alto):
    for j in range(ancho):
        blue[i,j][0] = img[i,j][0]
        green[i,j][1] = img[i,j][1] 
        red[i,j][2] = img[i,j][2] 
        
canales_imagen = np.concatenate((red,green,blue),axis=1)

cv2.imshow('imagen',canales_imagen)  
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
# Ejercicio N°16
# =============================================================================

# img = cv2.imread('perro.jpg')
# alto,ancho,dim = img.shape

# # Promedio

# for i in range(alto):
#     for j in range(ancho):
#         gris = (img[i,j][0] + img[i,j][1] + img[i,j][2])/3
#         img[i,j][0] = gris
#         img[i,j][1] = gris
#         img[i,j][2] = gris
        
# cv2.imshow('imagen',img) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Luma

# for i in range(alto):
#     for j in range(ancho):
#         gris = (img[i,j][0]*0.0722 + img[i,j][1]*0.7152 + img[i,j][2]*0.2126)/3
#         img[i,j][0] = gris
#         img[i,j][1] = gris
#         img[i,j][2] = gris
        
# cv2.imshow('imagen',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Desaturacion Min/Max

# for i in range(alto):
#     for j in range(ancho):
#         gris = (max(img[i,j][0],img[i,j][1],img[i,j][2]) + min(img[i,j][0],img[i,j][1],img[i,j][2]))/2
#         img[i,j][0] = gris
#         img[i,j][1] = gris
#         img[i,j][2] = gris
        
# cv2.imshow('imagen',img) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Unico canal

# img2 = np.full((alto,ancho,1),125,dtype = 'uint8')
# for i in range(alto):
#     for j in range(ancho):
#         gris = (img[i,j][0] + img[i,j][1] + img[i,j][2])/3
#         img2[i,j][0] = gris

# cv2.imshow('imagen',img2) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def escala_grises (imagen,retorno,metodo = 'c'):
    img = cv2.imread(imagen)
    alto,ancho,dim = img.shape
    
    if metodo == 'a': # Promedio
        for i in range(alto):
            for j in range(ancho):
                 gris = (img[i,j][0] + img[i,j][1] + img[i,j][2])/3
                 img[i,j][0] = gris
                 img[i,j][1] = gris
                 img[i,j][2] = gris
        if retorno == 'disco':
            return(cv2.imwrite('imagen_grises.jpg',img))
        if retorno == 'imagen':
            return(cv2.imshow('imagen',img))
        
    if metodo == 'b': # LUMA
        for i in range(alto):
            for j in range(ancho):
                gris = (img[i,j][0]*0.0722 + img[i,j][1]*0.7152 + img[i,j][2]*0.2126)/3
                img[i,j][0] = gris
                img[i,j][1] = gris
                img[i,j][2] = gris        
        if retorno == 'disco':
            return(cv2.imwrite('imagen_grises.jpg',img))
        if retorno == 'imagen':
            return(cv2.imshow('imagen',img))   

    if metodo == 'c': # Desaturacion Min/Max
        for i in range(alto):
            for j in range(ancho):
                gris = (max(img[i,j][0],img[i,j][1],img[i,j][2]) + min(img[i,j][0],img[i,j][1],img[i,j][2]))/2
                img[i,j][0] = gris
                img[i,j][1] = gris
                img[i,j][2] = gris   
        if retorno == 'disco':
            return(cv2.imwrite('imagen_grises.jpg',img))
        if retorno == 'imagen':
            return(cv2.imshow('imagen',img))   
        
    if metodo == 'd': # Unico canal
        img2 = np.full((alto,ancho,1),125,dtype = 'uint8')
        for i in range(alto):
            for j in range(ancho):
                gris = (img[i,j][0] + img[i,j][1] + img[i,j][2])/3
                img2[i,j][0] = gris       
        if retorno == 'disco':
            return(cv2.imwrite('imagen_grises.jpg',img2))
        if retorno == 'imagen':
            return(cv2.imshow('imagen',img2))  
    
escala_grises('perro.jpg', 'disco','c') # NUNCA ME FUNCIONO VER POR PANTALLA LA IMAGEN DESDE UNA FUNCION, SE CRASHEA EL PROGRAMA

# =============================================================================
# Ejercicio N°17
# =============================================================================

img = cv2.imread('perro.jpg')
alto,ancho,dim = img.shape

# SEPIA

for i in range(alto):
    for j in range(ancho):
        b = img[i,j][0]
        g = img[i,j][1]
        r = img[i,j][2]
        img[i,j][0] = r*0.272 + g*0.534 + b*0.131
        img[i,j][1] = r*0.349 + g*0.686 + b*0.168
        img[i,j][2] = r*0.393 + g*0.769 + b*0.189
        
cv2.imshow('imagen',img)
cv2.waitKey(0)
cv2.destroyAllWindows()        

