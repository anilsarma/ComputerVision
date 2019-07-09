import cv2
import os
import numpy as np
import imutils
import time
#from os import listdir
#from os.path import isfile, join
mypath="thermal"
imagePaths = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
knownEncodings = []
knownNames = []
def nothing(x):
    print(x)
    pass
cv2.namedWindow('image')


name = "anil_sarma"

lower_range = np.array([0, 0, 0], dtype=np.uint8)
upper_range = np.array([37, 232, 255], dtype=np.uint8)



cv2.createTrackbar('r','image',lower_range[2],255,nothing)
cv2.createTrackbar('g','image',lower_range[1],255,nothing)
cv2.createTrackbar('b','image',lower_range[0],255,nothing)

cv2.createTrackbar('R','image',upper_range[2],255,nothing)
cv2.createTrackbar('G','image',upper_range[1],255,nothing)
cv2.createTrackbar('B','image',upper_range[0],255,nothing)

orig = None
for (index, imagePath) in enumerate(imagePaths):
    print(imagePath)

    image = cv2.imread(imagePath)
    orig = image.copy()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)


    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    #lower_blue = np.array([110, 50, 50])
    #upper_blue = np.array([130, 255, 255])

    hm = cv2.applyColorMap(image.copy(), cv2.COLORMAP_JET)

    #cv2.imshow("original - ", image.copy())
    mask = cv2.inRange(hsv.copy(), lower_range, upper_range)
    #gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #mask = cv2.dilate(mask, None, iterations=2)
    #cv2.imshow("mask - ", mask.copy())

    cnts = cv2.findContours(mask.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for ctr in cnts:
        if cv2.contourArea(ctr) < 600:
            continue
        print("min area", cv2.contourArea(ctr))
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(ctr)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv2.imshow("table - ", image.copy())
    #break

r = 50
b = 50
g = 60
R = 255
B = 100
G = 100

while True:
    time.sleep(1/100)
    #cv2.imshow("original - ", image.copy())
    r = cv2.getTrackbarPos('r', 'image')
    g = cv2.getTrackbarPos('g', 'image')
    b = cv2.getTrackbarPos('b', 'image')

    R = cv2.getTrackbarPos('R', 'image')
    B = cv2.getTrackbarPos('B', 'image')
    G = cv2.getTrackbarPos('G', 'image')

    lower_range = np.array([b, g, r], dtype=np.uint8)
    upper_range = np.array([B, G, R], dtype=np.uint8)
    print(lower_range, upper_range)
    mask = cv2.inRange(hsv.copy(), lower_range, upper_range)
    # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    image = orig.copy()
    for ctr in cnts:
        if cv2.contourArea(ctr) < 600:
            continue
        #print("min area", cv2.contourArea(ctr))
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(ctr)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv2.imshow("table - ", image.copy())

    print("Shape", mask.shape, image.shape)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = imutils.resize(mask.copy(), width=700)
    image = imutils.resize(image.copy(), width=700)
    cv2.imshow("image", np.hstack( [mask, image] ))

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
#cap.release()
cv2.destroyAllWindows()
