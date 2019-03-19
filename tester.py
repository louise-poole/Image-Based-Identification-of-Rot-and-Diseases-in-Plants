import numpy as np 
import cv2
from matplotlib import pyplot as plt

def apply_invert(frame):
    return cv2.bitwise_not(frame)

img = cv2.imread('0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.jpg')

#cv2.imshow('image', img)
#cv2.imshow('invert', apply_invert(img))
#cv2.imshow('grayscale', cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.imshow('denoised', cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21))

#---------------------THRESHOLDING-------------------
thr = 127
ret,thresh1 = cv2.threshold(img,thr,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,thr,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,thr,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,thr,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,thr,255,cv2.THRESH_TOZERO_INV)

thresh = ['img','thresh1','thresh2','thresh3','thresh4','thresh5']
 
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(eval(thresh[i]),'gray')
    plt.title(thresh[i])
 
plt.show()
#------------------------------------------------------

#-------------------ADAPTIVE THRESHOLDING----------------
'''img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img,5)
 
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
 
plt.subplot(2,2,1),plt.imshow(img,'gray')
plt.title('input image')
plt.subplot(2,2,2),plt.imshow(th1,'gray')
plt.title('Global Thresholding')
plt.subplot(2,2,3),plt.imshow(th2,'gray')
plt.title('Adaptive Mean Thresholding')
plt.subplot(2,2,4),plt.imshow(th3,'gray')
plt.title('Adaptive Gaussian Thresholding')
 
plt.show()'''
#----------------------------------------------------

#---------------HISTOGRAM: BACKPROJECTION-----------------
''' #roi is the object or region of object we need to find
roi = cv2.imread('target.jpg')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
 
#target is the image we search in
target = img
hsvt = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
 
# calculating object histogram
imghist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
 
# normalize histogram and apply backprojection
cv2.normalize(imghist,imghist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],imghist,[0,180,0,256],1)
 
# Now convolute with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(dst,-1,disc,dst)
 
# threshold and binary AND
ret,thresh = cv2.threshold(dst,50,255,0)
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(target,thresh)
 
res = np.vstack((target,thresh,res))
cv2.imwrite('res.jpg',res)
cv2.imshow('image', res) '''
#-----------------------------------------------------

cv2.waitKey(0)
cv2.destroyAllWindows()