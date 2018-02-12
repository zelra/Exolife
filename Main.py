import numpy as np
import cv2
from numpy import vectorize
from numpy.lib.tests.test__datasource import valid_baseurl


def A1() :

    # Load an color image in grayscale
    img = cv2.imread('images/A1.pbm',0)

    #img = img + 100

    isize = img.shape
    maxi=0
    coord = []
    for i in range(0, isize[0]):
        for j in range (0, isize[1]):
            if img[i][j]>maxi:
                maxi=img[i][j]
                coord = [i, j]

    print(coord)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def A2() :

    img = cv2.imread('images/A2.pbm', 0)

    isize = img.shape
    valeurPixel=0
    pourcentagePixel = 0
    buffer =0
    pourcentageNuage = 0

    for i in range(0, isize[0]):
        for j in range (0, isize[1]):
                valeurPixel=img[i][j]
                pourcentagePixel = (valeurPixel/255)*100
                buffer = buffer + pourcentagePixel

    pourcentageNuage = buffer / img.size
    print (pourcentageNuage)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def A3() :

    img = cv2.imread('images/A3.pbm', 0)


    img1 =  cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2 )

    cv2.imshow('image', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def A4() :
    
    # Blending, opening, closing

    img = cv2.imread('images/A4.pbm', 0)
    img1 = cv2.imread('images/A4bis.pbm', 0)

    isize = img.shape
    isize1 = img1.shape
    px1 = 0
    px2 = 0
    px = 0

    for i in range(0, isize[0]):
        for j in range (0, isize[1]):
            px1 = img[i][j]
            px2 = img1[i][j]
            img[i][j] = min(px1, px2)

    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


    cv2.imshow('image', closing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def B1() :

# Egalisation, normalization
    img = cv2.imread('images/B1.pbm', 0)

    imgE = cv2.equalizeHist(img)

    b = cv2.normalize(imgE, imgE, 0, 255, cv2.NORM_MINMAX )

    cv2.imshow('image', imgE)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def B2() :
# Normalisation implémenté dans l'égalisation

    img = cv2.imread('images/B2.pbm', 0)

    kernel = np.ones((5, 5), np.uint8)

    b = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    closing = cv2.morphologyEx(b, cv2.MORPH_HITMISS, kernel)
    c = cv2.normalize(closing, closing, 0, 255, cv2.NORM_MINMAX)
    opening = cv2.morphologyEx(c, cv2.MORPH_CLOSE, kernel)
#    blur = cv2.bilateralFilter(opening,9,75,5)
#    cv2.threshold(opening, 50, 255, cv2.THRESH_BINARY , opening)
#    median = cv2.medianBlur(opening,5)

    cv2.imshow('image', opening)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def B3() :

B3()