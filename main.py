import cv2
import numpy as np
import math

from detekcijaLinija import detekcija_plave, detekcija_zelene, koeficijentiJednacine
from keras.datasets import mnist
from keras.utils import to_categorical
import os.path
from brojevi import detekcija_Regiona,podudaranje_Reg,razvrstavanje_brr,brr_zabiljezen
from keras.utils import np_utils
from neuronskaMreza import kreiraj_Model, obucavanjeMreze
from nekePomocne import otvaranje_slike,image_bin,sivilo 

lista_videa = ['video-0.avi', 'video-1.avi', 'video-2.avi', 'video-3.avi', 'video-4.avi',
          'video-5.avi', 'video-6.avi', 'video-7.avi', 'video-8.avi',  'video-9.avi']

ukupna_suma = []
naziv_videa = []
 
def ucitaj_sve():
    #ucitavanje svih videaa
    for p in range(10):
        putanja = 'video-' + str(p) + '.avi'
        print(putanja)
        naziv_videa.append(putanja)
        rezultat = ucitaj_jedan_video(putanja)
        print(rezultat)
        ukupna_suma.append(rezultat)
 
     
        
def ucitaj_jedan_video(putanja):
    #ucitavanje jednog videa
    okvir = 0
    video_capture = cv2.VideoCapture(putanja)
    #indeksiranje frejmova
    video_capture.set(1,okvir) 
    ret_val, orig_frame = video_capture.read()
    nn_model = kreiraj_Model()
    nn_model.load_weights(''
    'model.h5')
    print("Model je ucitan!")
   

    
    okvir_slike = otvaranje_slike(orig_frame)
    
    plaveLinije = detekcija_plave(okvir_slike) #pozivamo funkciju za detekciju plave linije
    zeleneLinije =detekcija_zelene(okvir_slike) #pozivamo funkciju za detekciju zelene linije
 
    a, b = koeficijentiJednacine(zeleneLinije) #funkcija za pronalazenje koeficijenata y= kx+n
    a1, b1 = koeficijentiJednacine(plaveLinije)

    print(plaveLinije)
    print(zeleneLinije)
    
    sumaP = 0 #suma ispod plave
    sumaZ = 0 #suma ispod zelene
    rst = 0
    brF = [] #broj okvira/frejmova brojeva
    konacan_iznos = 0
    
    while True:
        okvir+=1
        ret_val, orig_frame = video_capture.read()
        #provjeravamo da li je frejm zahvacen
        if not ret_val:
           break
        frejm = image_bin(sivilo(orig_frame)) #konvertovanje u sivu zbog lakseg prepoznavanja
        detektovan_reg, brReg, koord = detekcija_Regiona(orig_frame.copy(), frejm) #detektovan region
        coskovi = [] #coskovi regiona
        
        for c in koord:
            (x,y,w,h) = c
            coskovi.append([(x+w),(y+h)]) 
        if rst == 20:  #neke brojeve registruje vise putaa
            rst = 0
            brF = []
     
        nekiBr=0
        for cos in coskovi:
            cos1,cos2 = cos
            plavaLinija = podudaranje_Reg(plaveLinije, a1, b1, cos1, cos2) # provjeravamo da li se neki brojevi identifikuju vise putraa        
            if plavaLinija == True:                
                reg = brr_zabiljezen(cos,brF)
                if reg == False:
                    brrr = {'koordinate': (cos1, cos2)}
                    brF.append(brrr)
                    trenutni_Brr = razvrstavanje_brr([brReg[nekiBr]],alf,nn_model)
                    konacan_iznos+=trenutni_Brr
                    sumaP+=trenutni_Brr #sabiranje brojeva ispod plave linije
                    print('\\\\\PLAVA LINIJA/////')
                    print((trenutni_Brr))  #prikazivanje broja koji je identifikovan 
                    
            zelenaLinija = podudaranje_Reg(zeleneLinije, a, b, cos1, cos2)
            if zelenaLinija == True:                
                reg = brr_zabiljezen(cos,brF)
                if reg == False:
                    brrr = {'koordinate': (cos1, cos2)}
                    brF.append(brrr)
                    trenutni_Brr = razvrstavanje_brr([brReg[nekiBr]],alf,nn_model)
                    konacan_iznos-=trenutni_Brr
                    sumaZ+=trenutni_Brr #sabiranje brojeva ispod zelene linije
                    print('\\\\\ZELENA LINIJA/////')
                    print((trenutni_Brr))  #prikazivanje broja koji je identifikovan
            nekiBr+=1
        rst+=1
         
        frejm2 = cv2.GaussianBlur(orig_frame,(5,5),0)
        #prikaz suma za plavu i zelenu liniju, kao i prikaz konacne sume (razlika plave i zelene linije)
        cv2.putText(detektovan_reg, "Suma: " + str(sumaP), (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,(255, 0, 0), 1)
        cv2.putText(detektovan_reg, "Suma2: " + str(sumaZ), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,(0, 255, 0), 1)
        cv2.putText(detektovan_reg, "Konacna: " + str(sumaP-sumaZ), (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,(0, 0, 255), 1)

        cv2.imshow('video',detektovan_reg)  #prikazivanje videaa sa oznacenim brojevima
        key = cv2.waitKey(30)
        if key==27:
            break
    cv2.destroyAllWindows()    
    video_capture.release()   
    return konacan_iznos
  

def upisi_u_fajl(): #upisivanje u fajl
    fajl = open('out.txt', 'w')
    tekst = 'RA 161/2015 Sara Celik\nfile\tsum\n'
    for i in range(10):
        tekst += lista_videa[i] + '\t' + str(ukupna_suma[i])+'\n' #rezultati za svaki videoo


    fajl.write(tekst)
    fajl.close()
alf = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ucitaj_sve()
upisi_u_fajl()


    
    
    

    