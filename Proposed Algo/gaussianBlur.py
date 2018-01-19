import cv2
import numpy as np

def gaussianBlur(img, ksize, sigma):
    # Cette fonction est utilisee pour le filtre implemente de Gaussien
    nx = np.arange(-ksize, ksize+1, 1)
    ny = np.arange(-ksize, ksize+1, 1)
    x,y = np.meshgrid(nx,ny)
    expComp = -(x**2+y**2)/(2*sigma*sigma)
    kernel = np.exp(expComp)/(2*np.pi*sigma*sigma)
    M = np.size(x,0)-1
    N = np.size(y,0)-1
    if np.size(np.shape(img))==2:
        outputImg = np.empty((np.size(img,0),np.size(img,1)))
        I = np.lib.pad(img,ksize,'constant',constant_values=0)
        for i in range(np.size(I,0)-M):
            for j in range(np.size(I,1)-N):
                temp = np.dot(I[j:j+M+1,i:i+N+1],kernel)
                outputImg[j,i]=np.round(temp.sum())
    else:
        outputImg = np.empty((np.size(img,0),np.size(img,1),np.size(img,2)))
        I1 = np.lib.pad(img[0:np.size(img,0),0:np.size(img,1),0],ksize,'constant',constant_values=0)
        for i in range(np.size(I1,0)-M):
            for j in range(np.size(I1,1)-N):
                temp = np.dot(I1[i:i+M+1,j:j+N+1],kernel)
                outputImg[i,j,0]=np.round(temp.sum())
        I2 = np.lib.pad(img[0:np.size(img,0),0:np.size(img,1),1],ksize,'constant',constant_values=0)
        for i in range(np.size(I2,0)-M):
            for j in range(np.size(I2,1)-N):
                temp = np.dot(I2[i:i+M+1,j:j+N+1],kernel)
                outputImg[i,j,1]=np.round(temp.sum())
        I3 = np.lib.pad(img[0:np.size(img,0),0:np.size(img,1),2],ksize,'constant',constant_values=0)
        for i in range(np.size(I3,0)-M):
            for j in range(np.size(I3,1)-N):
                temp = np.dot(I3[i:i+M+1,j:j+N+1],kernel)
                outputImg[i,j,2]=np.round(temp.sum())
    outputImg = outputImg.astype(np.uint8)
    return outputImg

image = cv2.imread("F:/Projet fin d'Etudes/Image/IMG_0493.jpg",0)
out = gaussianBlur(image,3,3)
cv2.namedWindow("PicBlurred",cv2.WINDOW_NORMAL)
cv2.imshow("PicBlurred", out)
cv2.waitKey(0)
cv2.destroyAllWindows()