from audioop import reverse
import numpy as np
import cv2 as cv
import math

cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (11,11), 0)
    threshold = cv.threshold(gray, 170, 255, cv.THRESH_BINARY)
    edged = cv.Canny(threshold[1], 75, 200)
    lsd = cv.createLineSegmentDetector(0)

    lines = lsd.detect(threshold[1])[0]

    line_len = np.zeros(len(lines))
    for l in range(len(lines)):
        line_len[l] = math.sqrt( (lines[l][0][0] - lines[l][0][2] )**2 + (lines[l][0][1] - lines[l][0][3] )**2)
    dominant_len = -np.sort(-line_len)
    index_lines = np.argsort(-line_len)
    lines = lines[index_lines][:3]

    draw = lsd.drawSegments(edged, lines)
    # Display the resulting frame
    cv.imshow('frame', draw)
    print(len(lines))

    cnts = cv.findContours(edged, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(cnts[0], key = cv.contourArea, reverse = True)

    for c in cnts[0]:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.1*peri, True)
        screenCnt = approx
        if len(approx) == 4:
            screenCnt = approx
            break

    image = cv.drawContours(frame, screenCnt, -1, (0,255,0), 2)
    cv.imshow('processed', image)


    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()