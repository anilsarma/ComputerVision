import numpy as np
import cv2
import imutils
import time
cap = cv2.VideoCapture()
#front
addrs = [
    #0
]


W=400
class Data:
    def __init__(self, x):
        self.firstFrame = None
        self.frame = None
        self.x = x
        self.cap = cv2.VideoCapture(x)

    def open(self):
        if not self.cap.isOpened():
            self.cap.open(self.x)
db = {}

for x in addrs:
    data = Data(x)
    db[x] = data
    data.open()

def motion(key, frame):
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    data = db.get(key)
    data.frame = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('framwe', gray)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if data.firstFrame is None:
        data.firstFrame = gray
        return data


    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(data.firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 50:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    data.frame = frame
    return data

def capture(key):
    #print(cap)
    data = db.get(key)

    ret, frame = data.cap.read()
    if frame is None:
        print(key, ret)
        #data.cap.release()
        #data.open()
        return data.frame
    # height, width, depth = frame.shape
    # imgScale = W / width
    # newX, newY = frame.shape[1] * imgScale, frame.shape[0] * imgScale
    # frame = cv2.resize(frame, (int(newX), int(newY)))

    frame = motion(key, frame).frame
    return frame
old = []
while(True):
    time.sleep(1/100)
     # Capture frame-by-frame

    frame = []
    for cap in db.keys():
        try:
            c = capture(cap)
            frame.append(c);


        except Exception as e:
            print(e)
            frame.append(None)
            #raise(e)

    if len(old) == 0:
        old = frame

    for x in range(0, len(frame)):
         if frame[x] is None:
             frame[x] = old[x]
# Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    old = frame
    # Display the resulting frame
    cv2.imshow('frame',np.hstack(frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
