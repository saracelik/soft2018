# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:29:33 2019

@author: Sara
"""

import cv2
import numpy as np
from nekePomocne import skaliranje
import math

def transformacija_Reg(region):  #pomocna funkcija za transformaciju reg na sliku cije su dimenzije matrica 28x28
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

def detekcija_Regiona(slikaa, pic_bin): #funkcija za detekciju regiona
    pic, kons, hijerarhija = cv2.findContours(pic_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #pronalazimo konture brojaa
    lista_ss = []  #lista gdje cemo smjestiti detektovane regione
    rgg = [] 
    tjemena = [] #tjemena detektovanog regiona
       
    for kon_kk in kons:
        x, y, w, h = cv2.boundingRect(kon_kk)  
        povrsina = cv2.contourArea(kon_kk)
        if (povrsina > 15 and h < 45 and h > 13 and w > 1) or (povrsina > 15 and h < 45 and h > 9 and w > 12):
            oblast_rg = pic_bin[y:y + h + 1, x:x + w + 1]
            rgg.append([transformacija_Reg(oblast_rg), (x, y, w, h)])
            kk = (x, y, w, h)
            tjemena.append(kk)
            cv2.rectangle(slikaa, (x, y), (x + w, y + h), (0, 255, 0), 2)
    rgg = sorted(rgg, key=lambda item: item[1][0]) #sortiramo regionee
    lista_ss = lista_ss = [oblast_rg[0] for oblast_rg in rgg]
    tjemena = sorted(tjemena, key=lambda item: item[0])
    return slikaa, lista_ss, tjemena


def stim_Neuron(izlaz):  #stimulisani neuron na izlazu
    return max(enumerate(izlaz), key=lambda x: x[1])[0]


def prikazivanjeRez(izlazi, alf):
    rezz = []
    for izlaz in izlazi:
        rezz.append(alf[stim_Neuron(izlaz)])
    return rezz


def priprema_za_list_reg(ulazi): #skaliranje elemenata i smjestanje u vektor
    spreman = []
    for ulaz in ulazi:
        skaliran = skaliranje(ulaz)
        spreman.append(skaliran)

    return spreman


def konvertt(alf):
    izlazi = []
    blaa = len(alf)
    for index in range(len(alf)):
        izlaz = np.zeros(len(alf))
        izlaz[index] = 1
        izlazi.append(izlaz)
    return np.array(izlazi)


def podudaranje_Reg(tjemena,a,b,s1,s2): #funkcija za pronalazenje regiona koji se sudaraju
    x1, y1, x2, y2 = tjemena
    if x2+2>=s1>=x1-5 and y1+5>=s2>=y2-1:
        d = a*s1 + b
        if abs(int(s2)-int(d))<= 2: 
            return True
    return False


def razvrstavanje_brr(img, alf, model): #prepoznavanje brojevaa
    temp = np.array(img)
    
    inp = priprema_za_list_reg(temp)
    inp = np.expand_dims(inp, axis=3)
    rezz = model.predict(np.array(inp, np.float32))
    rez = prikazivanjeRez(rezz, alf)[0]
    return rez


def Identifikovan_brr(n, br, brojevi): #funkcija koja sluzi da ustanovimo koji je broj identifikovan
    ret = []
    for b in brojevi:
        udaljenost = distance(br['koordinate'], b['koordinate'])
        if udaljenost < n:
            ret.append(b)
    return ret


def brr_zabiljezen(n,brF):
    x,y = n
    brr = {'koordinate': (x, y)}
    akopostoji = Identifikovan_brr(10, brr, brF)
    proradi = len(akopostoji)
    if proradi == 0:
        return False
    else:
        return True 
#pomocne funkcije za vektore; duzina,udaljenost,skaliranje,dodavanje
def dot(v,w):
    x,y = v
    X,Y = w
    return (X*x+Y*y)
 
def length(v):
    x,y = v
    i=x*x
    j=y*y
    return i+j
  
def vector(b,e):
    x,y = b
    X,Y = e
    return X-x, Y-y
  
def unit(v):
    x,y = v
    mag = length(v)
    #s=x/mag
    #d=y/mag
    #return (s,d)
    return (x / mag, y / mag)
  
def distance(p0,p1):
    return length(vector(p0,p1))
  
def scale(v,sc):
    x,y = v
    a=x*sc
    b=y*sc
    return (a, b)
  
def add(v,w):
    x,y = v
    X,Y = w
    a=x+X
    b=y+Y
    return (a, b)
  
def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)      
    r = 1                       
    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = scale(line_vec, t)            
    dist = distance(nearest, pnt_vec)        
    nearest = add(nearest, start)                                          
    return (dist, (int(nearest[0]), int(nearest[1])), r)