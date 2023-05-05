import numpy as np
import cv2
#from pathlib import Path
import matplotlib.pyplot as plt

def importationMesures(nomDeTonFichierTexte):
    #nomDeTonFichierTexte = Path(nomDeTonFichierTexte)
    nomDeTonFichierTexte =nomDeTonFichierTexte.replace("\\" , "/")
    with open(nomDeTonFichierTexte, 'r',encoding="ISO-8859-1") as file:
        data = file.read()
        txt = data.replace(',', '.')
    with open('fichier2.txt','w',encoding="ISO-8859-1") as file:
        file.write(txt)
    file.close()
    t, x, y = np.loadtxt('fichier2.txt',delimiter='\t',skiprows=3, unpack=True)
    return t,x,y

def filtrage(vx,filtrageSize = 5) :
    vecteurMoyenne1 = np.ones(filtrageSize)/filtrageSize

    return np.correlate(vx, vecteurMoyenne1, "same")


def affichage(x,y,vx,vy,ax,ay,t,v,filtrageSize = 5):
    img = cv2.imread('img_eol.png')
    nbL =img.shape[0]
    nbC = img.shape[1]
    pixelSize = 0.282
    moydiv2 = max(1,int(filtrageSize/2))+2
    
    # AFFICHAGE DES VITESSES V :
    
    x_v =x[0+moydiv2:-2-moydiv2] 
    y_v =y[0+moydiv2:-2-moydiv2]
    x_v = metersToPixel(x_v,pixelSize)
    y_v = nbL-metersToPixel(y_v,pixelSize)
    x_v = x_v-min(x_v)+479
    y_v = y_v-min(y_v) + 74
    endPoint_vx = np.add(x_v,vx[0+moydiv2:-1-moydiv2]/10)
    endPoint_vy = np.add(y_v,-vy[0+moydiv2:-1-moydiv2]/10)
    
    for i in range(0,len(x_v)):
        cv2.arrowedLine(img, (int(x_v[i]),int(y_v[i])), (int(endPoint_vx[i]),int(endPoint_vy[i])),
                                             (0,0,255), 2)
    cv2.arrowedLine(img, (400,30), (430,30),
                                             (0,0,255), 2)
    cv2.putText(img, "vitesse", (435,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0,0,255), 1, cv2.LINE_AA)
    # AFFICHAGE DES accélération A :
    
    endPoint_ax = np.add(x_v,ax[0+moydiv2:-moydiv2]/3)
    endPoint_ay = np.add(y_v,-ay[0+moydiv2:-moydiv2]/3)
    for i in range(0,len(x_v)):
        cv2.arrowedLine(img, (int(x_v[i]),int(y_v[i])),
                        (int(endPoint_ax[i]),int(endPoint_ay[i])),
                                             (255,0,0), 2)
    cv2.arrowedLine(img, (400,60), (430,60),
                                             (255,0,0), 2)
    cv2.putText(img, "acceleration", (435,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,0,0), 1, cv2.LINE_AA)
    
    resized = img[:,400:900]
    #cv2.resize(img, (int(nbC/2),int(nbL/2)), interpolation = cv2.INTER_AREA)
    plt.figure(figsize=(16,10))
    plt.imshow(resized)
    plt.show()
    
    vmoy = np.mean(v[6:len(v)])
    plt.plot(t[6:-1],v[6:len(v)],label='v')
    plt.plot(t[6:-1],np.ones(len(v)-6)*vmoy,label='vitesse moyenne = '+ str(int(vmoy)) + ' m/s')
    
    plt.xlabel("temps (s)")
    plt.ylabel("vitesse (m/s)")
    plt.legend()
    plt.show()
    
def metersToPixel(v,pixelSize):
    pix = v/pixelSize
    return pix
    
    
    