import numpy as np
import cv2
import time
import csv

f = open("cl_save.csv","a")
csvWriter = csv.writer(f)
cap = cv2.VideoCapture(0)
start = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray, (5, 5), 1)
    circles = cv2.HoughCircles(gray1, cv2.HOUGH_GRADIENT, 1, 60,
                               param1=10, param2=85, minRadius=0, maxRadius=85)

    if circles is not None and start is None:
        start = time.time()
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(frame,(i[0],i[1]),i[2],(0,0,0),2)
            cv2.circle(frame,(i[0],i[1]),2,(0,0,0),3)   

    if circles is not None and start is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(frame,(i[0],i[1]),i[2],(0,0,0),2)
            cv2.circle(frame,(i[0],i[1]),2,(0,0,0),3) 

    elif circles is None and start is not None:      
        elapsed_time = time.time() - start



        print(elapsed_time)
        listData = []
        listData.append(str(elapsed_time))
        csvWriter.writerow(listData)
        start = None

    cv2.imshow('preview', frame)

    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cv2.destroyAllWindows()