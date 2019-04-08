# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 18:05:45 2019

@author: Sara
"""

import cv2
import numpy as nmp
#neke pomocne funckije vezane za slike
def skaliranje(slika):  # skalira elemente slike sa 0-255 na opseg 0-1
    return slika / 255


def invertovanje(slika):
    return 255 - slika


def sivilo(slika): #cvtColor koristimo da bi sliku konvertovali u sivu radi lakseg rada
    return cv2.cvtColor(slika, cv2.COLOR_RGB2GRAY)


def transformacija_matrice(slika):   #funkcija za transformaciju matrice u vektor
    return slika.flatten()


def image_bin(slika):
    visina, sirina = slika.shape[0:2]
    ret, img_bin = cv2.threshold(slika, 127, 255, cv2.THRESH_BINARY)
    return img_bin

def image_erode(slika):
    kernel = nmp.ones((3,3)) 
    return cv2.erode(slika, kernel, iterations=1)

def prosirivanje(slika):
    kernel = nmp.ones((3,3))
    return cv2.dilate(slika, kernel, iterations=1)

def otvaranje_slike(okvir_slike):
    img_erode = image_erode(okvir_slike)
    slikaa = prosirivanje(img_erode)
    return slikaa
