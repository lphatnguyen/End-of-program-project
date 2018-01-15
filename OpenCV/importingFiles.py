import cv2
import os
from PIL import Image, ImageFilter
import time

path="F:/Projet fin d'Etudes/Image/"
filename=os.listdir(path)
time1 = time.time()
i=0

for string in filename:
    filename[i]=path+filename[i]
    image = cv2.imread(filename[i],-1)
    output = cv2.GaussianBlur(image,(3,3),0)
    cv2.imwrite("F:/Projet fin d'Etudes/Output OpenCV/image"+str(i)+".jpg",output)
    i=i+1
time1 = time.time()-time1

time2 = time.time()
i=0
for string in filename:
    #filename[i]=path+filename[i]
    image = Image.open(filename[i])
    output = image.filter(ImageFilter.GaussianBlur(1))
    output.save("F:/Projet fin d'Etudes/Output Pillow/image"+str(i)+".jpg")
    i=i+1

time2= time.time()-time2

print("OpenCV Gaussian Blur execution time : ", time1)
print("Pillow Gaussian Blur execution time : ", time2)