import numpy as np
import cv2
import time
import datetime
import imutils
sdThresh = 10
font = cv2.FONT_HERSHEY_SIMPLEX

# TODO: Face Detection 1
def distMap(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:, :, 0] ** 2 + diff32[:, :, 1] ** 2 + diff32[:, :, 2] ** 2) / np.sqrt(255 ** 2 + 255 ** 2 + 255 ** 2)
    dist = np.uint8(norm32 * 255)
    return dist


#cv2.namedWindow('frame')
#cv2.namedWindow('dist')

# capture video stream from camera source. 0 refers to first camera, 1 referes to 2nd and so on.
cap = cv2.VideoCapture(0)

_, frame1 = cap.read()
_, frame2 = cap.read()
cv2.imshow('frame', frame1)
cv2.namedWindow('dist')
facecount = 0
while(True):
    _, frame3 = cap.read()
    rows, cols, _ = np.shape(frame3)    
    cv2.imshow('dist', frame3)
    dist = distMap(frame1, frame3)

    frame1 = frame2
    frame2 = frame3

    # apply Gaussian smoothing
    mod = cv2.GaussianBlur(dist, (9,9), 0)

    # apply thresholding
    _, thresh = cv2.threshold(mod, 100, 255, cv2.THRESH_BINARY)

    # calculate st dev test
    _, stDev = cv2.meanStdDev(mod)

    cv2.imshow('dist', mod)


    cv2.putText(frame2, "Standard Deviation - {}".format(round(stDev[0][0],0)), (70, 70), font, 1, (255, 0, 255), 1, cv2.LINE_AA)


    if stDev > sdThresh:
        cv2.imshow('thresh', thresh)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cv2.imshow('thresh dialeed', thresh)
        print(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), ": Motion detected.. Do something!!!");
        #TODO: Face Detection 2
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(thresh.copy(), mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        framex = frame2.copy()
        count = 0
        for ctr in cnts:
            if cv2.contourArea(ctr) < 500:
                continue
            count +=1
            print("min area", cv2.contourArea(ctr))
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(ctr)
            cv2.rectangle(framex, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if count>0:
            cv2.imshow('motion', framex)

    cv2.imshow('frame', frame2)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
