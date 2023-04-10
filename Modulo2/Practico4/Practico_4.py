# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:24:19 2022

@author: jpman
"""

# Practico 4

# =============================================================================
# Ejercicio 1 
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns

# Funcion final para hacer histograma de una imagen en escala de grises 

def hist_gris (imagen,modo):
    lista_num = []
    if type(imagen) == str:
        img = cv2.imread(imagen)
    else:
        img = imagen
    alto,ancho,dim = img.shape
    for i in range(alto):
        for j in range(ancho):
            
            gris = (img[i,j][0]*0.0722 + img[i,j][1]*0.7152 + img[i,j][2]*0.2126)/3
            img[i,j][0] = gris
            img[i,j][1] = gris
            img[i,j][2] = gris 
            
            lista_num.append(img[i,j][1])
    if modo == 'kde':
        fig, kdeplot = plt.subplots(figsize = (8,6))
        kdeplot = sns.kdeplot(data = lista_num,fill = True)
        kdeplot.set_xlim(left=0, right=255)
    if modo == 'hist':
        fig, hist = plt.subplots(figsize = (8,6))
        hist = sns.histplot(x = lista_num, bins = 255, binrange = (0,255))
        hist.set_xlim(left=0, right=255)
        
    return(plt.show())      
    
hist_gris('spider.jpeg','hist')


# =============================================================================
# Ejercicio 2
# =============================================================================


def hist_color (imagen,color):
    red = []
    green = []
    blue = []
    if type(imagen) == str:
        img = cv2.imread(imagen)
    else:
        img = imagen
    alto,ancho,dim = img.shape
    for i in range(alto):
        for j in range(ancho):
            
            blue.append(img[i,j][0])
            green.append(img[i,j][1])
            red.append(img[i,j][2])
        
    if color == 'blue':   
        lista_num = blue
        fig, kdeplot = plt.subplots(figsize = (8,6))
        kdeplot = sns.kdeplot(data = lista_num,fill = True, color = 'blue')
        kdeplot.set_xlim(left=0, right=255)
        
    if color == 'green':
        lista_num = green
        fig, kdeplot = plt.subplots(figsize = (8,6))
        kdeplot = sns.kdeplot(data = lista_num,fill = True, color = 'green')
        kdeplot.set_xlim(left=0, right=255)
        
    if color == 'red':
        lista_num = red
        fig, kdeplot = plt.subplots(figsize = (8,6))
        kdeplot = sns.kdeplot(data = lista_num,fill = True, color = 'red')
        kdeplot.set_xlim(left=0, right=255)

       
    # fig, hist = plt.subplots(figsize = (8,6)) # Histograma comun 
    # hist = sns.histplot(x = lista_num, bins = 255, binrange = (0,255))
    # hist.set_xlim(left=0, right=255)
        
    return(plt.show())      
    
hist_color('spider.jpeg','blue')


# =============================================================================
# Ejercicio 3
# =============================================================================

# openCV || cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

# numpy || numpy.histogram(a, bins=10, range=None, density=None, weights=None)

# matplotlib || matplotlib.pyplot.hist(x, bins=None, range=None, density=False, weights=None, cumulative=False, 
#                                       bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, 
#                                       log=False, color=None, label=None, stacked=False, *, data=None, **kwargs)

# Funcion para hacer histograma RGB o BGR en realidad con openCV + matplotlib

def hist_rgb(imagen):
    if type(imagen) == str:
        img = cv2.imread(imagen)
    else:
        img = imagen
    bgr_planes = cv2.split(img)
    
    color = ('b','g','r')
    
    for i,col in enumerate(color):
        hist = cv2.calcHist(bgr_planes,[i],None,[256],[0,255])
        plt.plot(hist,color = col)
        plt.xlim([0,256])
    
    return(plt.show())

hist_rgb('spider.jpeg')

# =============================================================================
# Ejercicio 4
# =============================================================================


def bn (imagen,umbral):
    if type(imagen) == str:
        img = cv2.imread(imagen)
    else:
        img = imagen
    alto,ancho,dim = img.shape
    for i in range(alto):
        for j in range(ancho):
            if img[i,j][0] > umbral:
                img[i,j][0] = 255
            else:
                img[i,j][0] = 0
            if img[i,j][1] > umbral:
                img[i,j][1] = 255
            else:
                img[i,j][1] = 0
            if img[i,j][2] > umbral:
                img[i,j][2] = 255
            else:
                img[i,j][2] = 0 
    return(img)

cv2.imshow('imagen',bn('spider.jpeg',150))  # no encuentro la forma de poner el "imshow" dentro de la funcion y que ande.
cv2.waitKey(0)                              # No funcionan estos metodos de waitKey y destroyAllWindows y queda un recuadro gris
cv2.destroyAllWindows()                     # cuando meto el imshow dentro de la funcion


def bn2 (imagen,umbral):
    if type(imagen) == str:
        img = cv2.imread(imagen)
    else:
        img = imagen
    alto,ancho,dim = img.shape
    for i in range(alto):
        for j in range(ancho):
            gris = (img[i,j][0]*0.0722 + img[i,j][1]*0.7152 + img[i,j][2]*0.2126)/3
            img[i,j][0] = gris
            img[i,j][1] = gris
            img[i,j][2] = gris 
            
    for i in range(alto):
        for j in range(ancho):
            if img[i,j][0] > umbral:
                img[i,j][0] = 255
                img[i,j][1] = 255
                img[i,j][2] = 255
            else:
                img[i,j][0] = 0 
                img[i,j][1] = 0 
                img[i,j][2] = 0 
    return(img)

bn2('spider.jpeg',40)                      # La unica forma de que funcione es poniendo el waitKey y el destroyAllWindows de esta forma
cv2.waitKey(0)                              
cv2.destroyAllWindows()        

def bn3 (imagen,umbral):                    # igual que la anterior pero partiendo con una imagen en escala de grises
    if type(imagen) == str:
        img_bn3 = cv2.imread(imagen).copy()
    else:
        img_bn3 = imagen
    alto,ancho,dim = img_bn3.shape
            
    for i in range(alto):
        for j in range(ancho):
            if img_bn3[i,j][0] > umbral:
                img_bn3[i,j][0] = 255
                img_bn3[i,j][1] = 255
                img_bn3[i,j][2] = 255
            else:
                img_bn3[i,j][0] = 0 
                img_bn3[i,j][1] = 0 
                img_bn3[i,j][2] = 0 
    return(img_bn3)             


# =============================================================================
# Ejercicio 5
# =============================================================================

hist_gris('ciudad.jpg','hist')
# aproximadamente 65

bn2('ciudad.jpg',65)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
# Ejercicio 6
# =============================================================================


# Vamos a reciclar una funcion del practico anterior para convertir la imagen a escala de grises, con algunas modificaciones.

def imagen_a_gris (imagen,metodo = 'b'):
    img = cv2.imread(imagen)
    alto,ancho,dim = img.shape
    if metodo == 'a': # Promedio
        for i in range(alto):
            for j in range(ancho):
                 gris = int(int(img[i,j][0]) + int(img[i,j][1]) + int(img[i,j][2]))/3
                 img[i,j][0] = gris
                 img[i,j][1] = gris
                 img[i,j][2] = gris
        print("Imagen en escala de grises creada segun PROMEDIO")    
        return(img)
        
    if metodo == 'b': # LUMA
        for i in range(alto):
            for j in range(ancho):
                gris = int(int(img[i,j][0]*0.0722) + int(img[i,j][1]*0.7152) + int(img[i,j][2]*0.2126))/3
                img[i,j][0] = gris
                img[i,j][1] = gris
                img[i,j][2] = gris        
        print("Imagen en escala de grises creada segun LUMA")    
        return(img)   

    if metodo == 'c': # Desaturacion Min/Max
        for i in range(alto):
            for j in range(ancho):
                gris = int(int(max(img[i,j][0],img[i,j][1],img[i,j][2])) + int(min(img[i,j][0],img[i,j][1],img[i,j][2])))/2
                img[i,j][0] = gris
                img[i,j][1] = gris
                img[i,j][2] = gris   
        print("Imagen en escala de grises creada segun Desaturacion Min/Max")    
        return(img)
        
    if metodo == 'd': # Unico canal
        img2 = np.full((alto,ancho,1),125,dtype = 'uint8')
        for i in range(alto):
            for j in range(ancho):
                gris = int(int(img[i,j][0]) + int(img[i,j][1]) + int(img[i,j][2]))/3
                img2[i,j][0] = gris       
        print("Imagen en escala de grises creada segun Unico canal")    
        return(img2)  
    
imagen_gris = imagen_a_gris('actriz.png','b')

# cv2.imshow('imagen',imagen_gris)
# cv2.waitKey(0)
# cv2.destroyAllWindows()   

hist_gris(imagen_gris, 'hist')


# Quiza en 50     

mask = bn3(imagen_gris,48)

# =============================================================================
# Ejercicio 7
# =============================================================================

# cv2.imshow('imagen',mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def aplicar_mask (imagen,mask):
    if type(imagen) == str:
        img = cv2.imread(imagen)
    else:
        img = imagen
    alto,ancho,dim = img.shape
    for i in range(alto):
        for j in range(ancho):
            if mask[i,j][0] == 255:
                img[i,j][0] = 20
                img[i,j][1] = 117
                img[i,j][2] = 255
    return(img)

imagen_con_mask = aplicar_mask('actriz.png', mask)

cv2.imshow('imagen',imagen_con_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()  

# No fue el mejor resultado, quiza habria que jugar mas con la mascara y como se armo la imagen en escala de grises


# =============================================================================
# Ejercicio 8
# =============================================================================

img_gris = imagen_a_gris('fordt.png','b')
img_bn = bn(img_gris,45)

def inversa (imagen):
    if type(imagen) == str:
        img = cv2.imread(imagen)
    else:
        img = imagen
    alto,ancho,dim = img.shape
    img2 = np.full((alto,ancho,1),125,dtype = 'uint8')
    for i in range(alto):
        for j in range(ancho):
            if img[i,j][0] == 255:
                img2[i,j] = 0
            else:
                img2[i,j] = 255
    return(img2)  

inv = inversa(img_bn)

cv2.imshow('imagen',inv)
cv2.imshow('imagen2',img_bn)
cv2.waitKey(0)
cv2.destroyAllWindows()  

# =============================================================================
# Ejercicio 9
# =============================================================================

# Use las funciones que ya tenia definidas.

img_g = imagen_a_gris('actor.png','c')  # Primero pase la imagen a escala de grises

hist_rgb(img_g) # Despues con un histograma veo que valores usar para separar lo que quiero mas o menos

mask = bn3(img_g,110).copy()  # Despues hago un Thresholding

cv2.imshow('imagen',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
# Ejercicio 10
# =============================================================================

img8 = imagen_a_gris('fordt.png','d')

# img9 = cv2.copyMakeBorder(img8,1,1,1,1,cv2.BORDER_CONSTANT,0) # Para hacer un borde de 1 pixel

cv2.imshow('imagen',img8)
cv2.waitKey(0)
cv2.destroyAllWindows()

k1 = [[1,1,1],
      [1,1,1],
      [1,1,1]]

k2 = [[1,2,1],
      [2,4,2],
      [1,2,1]]

# sum(sum(k2,[]))  #  Para sumar todos los valroes de una matriz

for i in range(3):
    for j in range(3):
        print(k2[i][j])
        
print(range(10))

def blurring (imagen,k):
    n = sum(sum(k,[]))            
    if type(imagen) == str:
        img = cv2.imread(imagen).copy()
    else:
        img = imagen
    alto,ancho,dim = img.shape
    img_new = np.zeros((alto,ancho,3),np.uint8)
    for i in range(1,alto-1):
        for j in range(1,ancho-1):
            suma = 0
            for ki in range(3):
                for kj in range(3):
                    valor = img[i-ki-1,j-kj-1][0]*k[ki][kj]
                    suma = suma + valor
                    # print(suma/9)
            if int(suma/n) >=255:
                img_new[i,j] = 255
            else:
                img_new[i,j] = int(suma/n)
    return(img_new)
            
            
            
resultado = blurring(img8,k2)           
            
cv2.imshow('imagen',resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()    


# =============================================================================
# Ejercicio 11        
# =============================================================================

k3 = [[0,-1,0],
      [-1,4,-1],
      [0,-1,0]]

k4 = [[0,-1,0],
      [-1,5,-1],
      [0,-1,0]]

# n = 0
# for i in range(3):
#     for j in range(3):
#         n = n + abs(k3[i][j])
        
# print(n)

def sharpen (imagen,k):
    # n = sum(sum(k,[]))           
    if type(imagen) == str:
        img = cv2.imread(imagen).copy()
    else:
        img = imagen
    alto,ancho,dim = img.shape
    img_new = np.zeros((alto,ancho,3),np.uint8)
    for i in range(1,alto-1):
        for j in range(1,ancho-1):
            suma = 0
            for ki in range(3):
                for kj in range(3):
                    valor = img[i-ki-1,j-kj-1][0]*k[ki][kj]
                    suma = suma + valor
                    # print(suma/9)
            if int(suma) >=255:
                img_new[i,j] = 255
            if int(suma) <= 0:
                img_new[i,j] = 0
            else:
                img_new[i,j] = int(suma)
    return(img_new)    
    
                     
resultado2 = sharpen(img8,k4)           
            
cv2.imshow('imagen',resultado2)
cv2.waitKey(0)
cv2.destroyAllWindows()            

# =============================================================================
# Ejercicio 12
# =============================================================================


# Pense que deberia darle importancia a solo los pixeles de un vertice por decirlo 
# de algun modo, por ello diseñe el sigueinte kernel

kr0 = [[0,0,0],
       [0,1,1],
       [0,1,1]] 

# El resultado fue horrible, por lo que invesitgue en internet y encontre algo similar
# a esto. Que hay que darle importancia de un lado y restarle del otro. Por ello, 
# los resultados finales de kernels quedaron algo asi. 

kr1 = [[-2,-1,0],
       [0,1,0],
       [0,1,2]]

kr2 = [[0,-1,-2],
       [0,1,0],
       [2,1,0]]

 
            

resultado3 = sharpen(img8,kr1)           
            
cv2.imshow('imagen',resultado3)
cv2.waitKey(0)
cv2.destroyAllWindows()        


# =============================================================================
# Ejercicio 13
# =============================================================================
     
kb5 = [[1,4,7,4,1],
       [4,16,26,16,4],
       [7,26,41,26,7],
       [4,16,26,16,4],
       [1,4,7,4,1]]

def blurring5 (imagen,k):
    n = sum(sum(k,[]))            
    if type(imagen) == str:
        img = cv2.imread(imagen).copy()
    else:
        img = imagen
    alto,ancho,dim = img.shape
    img_new = np.zeros((alto,ancho,1),np.uint8)
    for i in range(1,alto-1):
        for j in range(1,ancho-1):
            suma = 0
            for ki in range(5):
                for kj in range(5):
                    valor = img[i-ki-1,j-kj-1][0]*k[ki][kj]
                    suma = suma + valor
                    # print(suma/9)
            if int(suma/n) >=255:
                img_new[i,j] = 255
            else:
                img_new[i,j] = int(suma/n)
    return(img_new)
            
            
            
            
resultado4 = blurring5(img8,kb5)           
            
cv2.imshow('imagen',resultado4)
cv2.waitKey(0)
cv2.destroyAllWindows()        

# Quiza el resultado es un tanto mejor, pero poco se puede apreciar. Lo que si 
# podemos notar mas facilmente es que demora cerca del triple en ejecutarse

# =============================================================================
# Ejercicio 14
# =============================================================================

def blurringbr (imagen,k):
    n = sum(sum(k,[]))            
    if type(imagen) == str:
        img = cv2.imread(imagen).copy()
    else:
        img = imagen
    alto,ancho,dim = img.shape
    img_new = np.zeros((alto,ancho,3),np.uint8)
    for d in range(3):
        for i in range(1,alto-1):
            for j in range(1,ancho-1):
                suma = 0
                for ki in range(3):
                    for kj in range(3):
                        valor = img[i-ki-1,j-kj-1][d]*k[ki][kj]
                        suma = suma + valor
                        # print(suma/9)
                if int(suma/n) >=255:
                    img_new[i,j,d] = 255
                else:
                    img_new[i,j,d] = int(suma/n)
    return(img_new)

# Solo fue necesario agregar un bucle mas, para darle el canal. 0 , 1 , 2 para BGR

resultado5 = blurringbr('fordt.png',k2)           
            
cv2.imshow('imagen',resultado5)
cv2.waitKey(0)
cv2.destroyAllWindows()   

# =============================================================================
# Ejercicio 15 
# =============================================================================

# Decidi separarlo en 3, pero podria hacerse una sola funcion en la cual se pueda
# seleccionar la opcion que quiera llevarse a cabo. Quiza dejar los kernels fijos
# dentro de la misma, y mediante las letras "a" , "b" , "c" el usuario pueda 
# seleccionar si hacer una deteccion con laplace, sobel o sobel suavizado.

klb = [[-1,-1,-1],
       [-1,8,-1],
       [-1,-1,-1]]

ks1 = [[1,2,1],
       [0,0,0],
       [-1,-2,-1]]

ks2 = [[-1,0,1],
       [-2,0,2],
       [-1,0,1]]


img_gris = imagen_a_gris('perrito.jpg','d')

# ------------------ Deteccion de bordes mediante Laplace --------------------

def bordes (imagen,k):
    # n = sum(sum(k,[]))           
    if type(imagen) == str:
        img = cv2.imread(imagen).copy()
    else:
        img = imagen
    alto,ancho,dim = img.shape
    img_new = np.zeros((alto,ancho,1),np.uint8)
    for i in range(1,alto-1):
        for j in range(1,ancho-1):
            suma = 0
            for ki in range(3):
                for kj in range(3):
                    valor = img[i-ki-1,j-kj-1][0]*k[ki][kj]
                    suma = suma + valor
                    # print(suma/9)
            if int(suma) >=255:
                img_new[i,j] = 255
            if int(suma) <= 0:
                img_new[i,j] = 0
            else:
                img_new[i,j] = int(suma)
    return(img_new)    

resultado6 = bordes(img_gris,klb)           
            
cv2.imshow('imagen',resultado6)
cv2.waitKey(0)
cv2.destroyAllWindows()   

# ------------------ Deteccion de bordes mediante Sobel  ---------------------

def bordes2 (imagen,k1,k2):
    # n = sum(sum(k,[]))           
    if type(imagen) == str:
        img = cv2.imread(imagen).copy()
    else:
        img = imagen
    alto,ancho,dim = img.shape
    img_new = np.zeros((alto,ancho,1),np.uint8)
    img_new_2 = np.zeros((alto,ancho,1),np.uint8)
    for i in range(1,alto-1):
        for j in range(1,ancho-1):
            suma = 0
            for ki in range(3):
                for kj in range(3):
                    valor = img[i-ki-1,j-kj-1][0]*k1[ki][kj]
                    suma = suma + valor
                    # print(suma/9)
            if int(suma) >=255:
                img_new[i,j] = 255
            if int(suma) <= 0:
                img_new[i,j] = 0
            else:
                img_new[i,j] = int(suma)
    for i in range(1,alto-1):
        for j in range(1,ancho-1):
            suma = 0
            for ki in range(3):
                for kj in range(3):
                    valor = img_new[i-ki-1,j-kj-1][0]*k2[ki][kj]
                    suma = suma + valor
                    # print(suma/9)
            if int(suma) >=255:
                img_new_2[i,j] = 255
            if int(suma) <= 0:
                img_new_2[i,j] = 0
            else:
                img_new_2[i,j] = int(suma)
    return(img_new_2)    

resultado7 = bordes2(img_gris,ks1,ks2)           
            
cv2.imshow('imagen',resultado7)
cv2.waitKey(0)
cv2.destroyAllWindows()  

# ------------ Deteccion de bordes mediante Sobel + suavizado  ----------------

def bordes3 (imagen,k,k1,k2):
    n = sum(sum(k,[]))           
    if type(imagen) == str:
        img = cv2.imread(imagen).copy()
    else:
        img = imagen
    alto,ancho,dim = img.shape
    img_new = np.zeros((alto,ancho,1),np.uint8)
    img_new_2 = np.zeros((alto,ancho,1),np.uint8)
    img_new_3 = np.zeros((alto,ancho,1),np.uint8)
    
    for i in range(1,alto-1):
        for j in range(1,ancho-1):
            suma = 0
            for ki in range(3):
                for kj in range(3):
                    valor = img[i-ki-1,j-kj-1][0]*k[ki][kj]
                    suma = suma + valor
                    # print(suma/9)
            if int(suma/n) >=255:
                img_new[i,j] = 255
            else:
                img_new[i,j] = int(suma/n)
                
    for i in range(1,alto-1):
        for j in range(1,ancho-1):
            suma = 0
            for ki in range(3):
                for kj in range(3):
                    valor = img_new[i-ki-1,j-kj-1][0]*k1[ki][kj]
                    suma = suma + valor
                    # print(suma/9)
            if int(suma) >=255:
                img_new_2[i,j] = 255
            if int(suma) <= 0:
                img_new_2[i,j] = 0
            else:
                img_new_2[i,j] = int(suma)
                
    for i in range(1,alto-1):
        for j in range(1,ancho-1):
            suma = 0
            for ki in range(3):
                for kj in range(3):
                    valor = img_new_2[i-ki-1,j-kj-1][0]*k2[ki][kj]
                    suma = suma + valor
                    # print(suma/9)
            if int(suma) >=255:
                img_new_3[i,j] = 255
            if int(suma) <= 0:
                img_new_3[i,j] = 0
            else:
                img_new_3[i,j] = int(suma)
    return(img_new_3)    

resultado8 = bordes3(img_gris,k2,ks1,ks2)           
            
cv2.imshow('imagen',resultado8)
cv2.waitKey(0)
cv2.destroyAllWindows()  


# =============================================================================
#  Ejercicio 16
# =============================================================================

a = (85,85,85)
b = (150,150,150)

def mask (imagen,a,b):      
    if type(imagen) == str:
        img = cv2.imread(imagen).copy()
    else:
        img = imagen
    alto,ancho,dim = img.shape
    img_new = np.zeros((alto,ancho,1),np.uint8)
    for d in range(3):
        for i in range(1,alto):
            for j in range(1,ancho):
                if all(img[i,j]>a):
                    if all(img[i,j]<b):
                        img_new[i,j] = (255)
                    else:
                        img_new[i,j] = (0)  
                else:
                    img_new[i,j] = (0) 
    return(img_new)

# Solo fue necesario agregar un bucle mas, para darle el canal. 0 , 1 , 2 para BGR

resultado10 = mask('fordt.png',a,b)           
            
cv2.imshow('imagen',resultado10)
cv2.waitKey(0)
cv2.destroyAllWindows()   
# =============================================================================
#  Ejercicio 17
# =============================================================================

def rotate (imagen,d):
    if type(imagen) == str:
        img = cv2.imread(imagen).copy()
    else:
        img = imagen
    alto,ancho,dim = img.shape
    if d == 270:
        img_new = np.zeros((ancho,alto,3),np.uint8)
        for d in range(3):  
            for i in range(1,alto):
                for j in range(1,ancho):
                    img_new[j,i][d] = img[i,j][d]
    if d == 90:
        img_new = np.zeros((ancho,alto,3),np.uint8)
        for d in range(3):  
            for i in range(1,alto):
                for j in range(1,ancho):
                    img_new[ancho-j,alto-i][d] = img[i,j][d]
    if d == 180:
        img_new = np.zeros((alto,ancho,3),np.uint8)
        for d in range(3):  
            for i in range(1,alto):
                for j in range(1,ancho):
                    img_new[alto-i,ancho-j][d] = img[i,j][d]
    return(img_new)

resultado9 = rotate('fordt.png',270)           
            
cv2.imshow('imagen',resultado9)
cv2.waitKey(0)
cv2.destroyAllWindows()  

# =============================================================================
# Ejercicio 18
# =============================================================================
            
def rotate (imagen,eje):
    if type(imagen) == str:
        img = cv2.imread(imagen).copy()
    else:
        img = imagen
    alto,ancho,dim = img.shape
    img_new = np.zeros((alto,ancho,3),np.uint8)
    if eje == "horizontal":
        for d in range(3):  
            for i in range(1,alto):
                for j in range(1,ancho):
                    img_new[i,j][d] = img[i,ancho-j][d]
    if eje == "vertical":
        for d in range(3):  
            for i in range(1,alto):
                for j in range(1,ancho):
                    img_new[i,j][d] = img[alto-i,j][d]
    
    return(img_new)

resultado = rotate('fordt.png',"vertical")           
            
cv2.imshow('imagen',resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()  