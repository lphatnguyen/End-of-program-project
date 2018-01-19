import numpy as np
import cv2
import time

def getGaussKernel(ksize, sigma):
    ax = np.arange(-ksize//2+1.,ksize//2+1.)
    xx,yy=np.meshgrid(ax,ax)
    kernel=np.exp(-(xx**2+yy**2)/(2.*sigma**2))
    kernel=kernel/np.sum(kernel)
    return kernel

def convo2d(image, kernel):
    m,n = kernel.shape
    y,x = image.shape
    img = np.lib.pad(image,m//2,'constant',constant_values=0)
    output = np.zeros((y,x))
    for i in range(y):
        for j in range(x):
            output[i][j] = np.sum(img[i:i+m, j:j+m]*kernel)
    output = np.uint8(output)
    return output

img = cv2.imread("F:/Projet fin d'Etudes/Output OpenCV/image3.jpg",0)
time1 = time.time()
ker = getGaussKernel(15,7)
#ker = np.ones((15,15))
#ker = ker/ker.sum()
outputImage = convo2d(img,ker)
time1 = time.time() - time1
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.imshow("Output", outputImage)
cv2.waitKey(0)
cv2.destroyAllWindows()