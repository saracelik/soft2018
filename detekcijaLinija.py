# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 19:51:37 2019

@author: Sara
"""

import cv2
import numpy as nmp
import math
from nekePomocne import sivilo


def detekcija_plave(slika):
    okvir_plave = slika.copy()
    
    okvir_plave[:, :, 1] = 0
    siva = sivilo(okvir_plave) #pozvali smo pomocnu funkciju za konverovanje u sivu boju radi lakseg prepoznavanja linije
    ret, hough = cv2.threshold(siva, 20, 255, cv2.THRESH_BINARY)
    #pronalazenje linije Hough transformacijom
    plava_linija = cv2.HoughLinesP(hough,1,nmp.pi/180,50,None,50,100)
    # ivice=[(0,0),(0,0)]
    #duz = 0
    a1 = min(plava_linija[:, 0, 0])
    b1 = max(plava_linija[:, 0, 1])
    a2 = max(plava_linija[:, 0, 2])
    b2 = min(plava_linija[:, 0, 3])
    
    tjemena_plave = (a1,b1,a2,b2) #koordinate plave linije na videu
                
    #return ivice

    return tjemena_plave

def detekcija_zelene(slika):
    okvir_zelene = slika.copy()
    
    okvir_zelene[:, :, 0] = 0
    siva = sivilo(okvir_zelene)
    ret, hough = cv2.threshold(siva, 20, 255, cv2.THRESH_BINARY)
    #pronalazenje linije Hough transformacijom
    zelena_linija = cv2.HoughLinesP(hough,1,nmp.pi/180,50,None,50,100)

    a1 = min(zelena_linija[:, 0, 0])
    b1 = max(zelena_linija[:, 0, 1])
    a2 = max(zelena_linija[:, 0, 2])
    b2 = min(zelena_linija[:, 0, 3])
    
    tjemena_zelene = (a1,b1,a2,b2)
    
    return tjemena_zelene
#prava linija se moze zapisati kao y=mx+b sto je polinom stepena 1
def koeficijentiJednacine(niz): #funkcija koja sluzi za pronalazak koeficijenata jednacine y=mx+b
    a1, b1, a2, b2 = niz
    a = [a1, a2]
    b = [b1, b2]
    koeficijenti = nmp.polyfit(a, b, 1) #pravljenje polinoma i vracanje koeficijanata
    return  koeficijenti[0],  koeficijenti[1]



   
   
    