import cv2
#import numpy as np

cap = cv2.VideoCapture("C:/Users/Luong Phat/Documents/projet fin d'etudes/traffic1.avi")
refFrame = None
img_area = None
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#outputVideo = cv2.VideoWriter("F:/Projet fin d'Etudes/Output OpenCV/output.avi",fourcc,20.0,(640,480))
while True:
    ret, frame = cap.read()
    
    if ret is False:
        break
    else:
        if refFrame is None:
            refFrame = frame
            refFrame = cv2.cvtColor(refFrame, cv2.COLOR_BGR2GRAY)
            img_area = refFrame.shape[0]*refFrame.shape[1]
            continue
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        outFrame = cv2.absdiff(refFrame,grayFrame)
        retval,thresh = cv2.threshold(outFrame, 15, 255, cv2.THRESH_BINARY)
        filteredFrame = cv2.medianBlur(thresh,15)
        (_,contour,_)=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in contour:
            contArea = cv2.contourArea(i)
            if (contArea> 0.001*img_area) and (contArea < 0.03*img_area):
                (x,y,w,h) = cv2.boundingRect(i)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    #outputVideo.write(frame)
        cv2.imshow("Output frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
        cv2.waitKey(1)