import numpy as np
import matplotlib.pyplot as plt
import cv2

capF = cv2.VideoCapture(0)
capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(2)

capF.set(3, 1280)
capF.set(4, 720)
capL.set(3, 1280)
capL.set(4, 720)
capR.set(3, 1280)
capR.set(4, 720)

# Skin Color Calibration

n = 0
dataF = np.zeros((3,90))
dataL = np.zeros((3,90))
dataR = np.zeros((3,90))

while True:
    if n < 150:
        cap = capF
        data = dataF
    elif n < 300:
        cap = capL
        data = dataL
    elif n < 450:
        cap = capR
        data = dataR
    else:
        cv2.destroyAllWindows()
        break

    ret, frame = cap.read()

    if (n % 150) < 30:
        cv2.circle(frame, (640,360), 5, (0,0,255), 2)
    elif (n % 150) < 60:
        cv2.circle(frame, (640,360), 5, (0,255,255), 2)
    else:
        data[:,(n%150)-60] = frame[360,640]
        b = data[0,(n%150)-60]
        g = data[1,(n%150)-60]
        r = data[2,(n%150)-60]
        #cv2.circle(frame, (640,360), 5, (0,255,0), 2)
        cv2.circle(frame, (640,360), 5, (b,g,r), 2)

    if n < 150:
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        disp = cv2.flip(frame, 1)
    elif n < 300:
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        disp = cv2.transpose(frame)
    elif n < 450:
        frame = cv2.flip(frame, 0)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        disp = cv2.transpose(frame)

    cv2.imshow('frame', disp)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    n += 1

capF.release()
capL.release()
capR.release()

np.save('dataF', dataF)
np.save('dataL', dataL)
np.save('dataR', dataR)

cv2.destroyAllWindows()
