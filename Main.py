import numpy as np
import cv2
from matplotlib import pyplot as plt
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
    #Multi-tresholding
    img = cv2.imread('images/B3.pbm', 1)
    imgG = cv2.imread('images/B3.pbm', 0)
    isize = img.shape

    maxi = 0
    min = 0
    moyenne = 0
    buffer = 0
    for i in range(0, isize[0]):
        for j in range(0, isize[1]):
            if imgG[i][j] > maxi:
                maxi = imgG[i][j]
            if imgG[i][j] < min:
                min = imgG[i][j]
#            buffer = buffer + imgG[i][j]
#    moyenne = buffer/(isize[0]* isize[1])
    moyenne = (maxi + min)/2
    print (min ,(min + moyenne)/2 ,moyenne, (maxi + moyenne)/2, maxi)

    for i in range(0, isize[0]):
        for j in range(0, isize[1]):
            if (imgG[i][j] < (min + moyenne)/2) :
                img[i][j] = [0, 0, 0]
            elif (imgG[i][j] > (min + moyenne)/2 and imgG[i][j] < moyenne) :
                img[i][j] = [0, 0, 255]
            elif (imgG[i][j] > moyenne and imgG[i][j] < (maxi + moyenne)/2) :
                img[i][j] = [0, 255, 0]
            elif (imgG[i][j] > (maxi + moyenne)/2 and imgG[i][j] < maxi) :
                img[i][j] = [255, 0, 0]

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def X1() :

    img = cv2.imread('images/X1.jpg', 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(133), plt.imshow(img_back, cmap='gray')
    plt.title('Output Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def X2() :
    #dct (demandé par david), mediant blurs

    img = cv2.imread('images/X2.pbm', 0)
#    imf = np.float32(img) / 255.0  # float conversion/scale

    median = cv2.medianBlur(img,3)
    median1 = cv2.medianBlur(median, 3)
    
#    test = cv2.medianBlur(median, 3)
#    dst = cv2.dct(median)  # the dct
#    final = cv2.idct(dst)   # convert back

    cv2.imshow('image', median1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def U1() :

    img = cv2.imread('images/U1.pbm', 0)

  #  imgG = cv2.GaussianBlur(img, (3, 3), 0)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)

    plt.subplot(2, 5, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 5, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 5, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 5, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 5, 5), plt.imshow(sobelxy, cmap='gray')
    plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])

    plt.show()

U1()