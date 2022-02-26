import cv2
import time
import sys
import imutils
from imutils.video import VideoStream
import math
import numpy as np
import matplotlib.pyplot as plt

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
arucoParams = cv2.aruco.DetectorParameters_create()

def calc_line_length(point1,point2):
    line_length = math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    return line_length

vs = VideoStream(src=0).start()
time.sleep(2.0)

fig = plt.figure()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)

    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    if len(corners) > 0:
        ids = ids.flatten()

        marker_sides = np.zeros((4,1,4))
        line_deltas = np.zeros((4))

        fig.clear()
        plt.ylim([0, 30])

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            topLine_length = int(calc_line_length(topLeft,topRight))
            rightLine_length = int(calc_line_length(topRight,bottomRight))
            bottomLine_length = int(calc_line_length(bottomLeft,bottomRight))
            leftLine_length = int(calc_line_length(topLeft,bottomLeft))

            marker_sides[markerID] = (topLine_length,rightLine_length,bottomLine_length,leftLine_length)

            line_averages = ((topLine_length+bottomLine_length)/2 , (leftLine_length+rightLine_length)/2)

            line_deltas[markerID] = math.sqrt((line_averages[0]-line_averages[1])**2)

            if np.any(line_deltas > max(line_averages)*0.05):
                g = 0
                r = 255
            else:
                r = 0
                g = 255

            cv2.line(frame, topLeft, topRight, (0, g, r), 2)
            cv2.line(frame, topRight, bottomRight, (0, g, r), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, g, r), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, g, r), 2)

            # compute and draw the center (x, y)-coordinates of the
            # ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

            # draw the ArUco marker ID on the frame
            cv2.putText(frame, str(markerID),
            (topLeft[0], topLeft[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2)

            x_label = ids
            x_space = np.arange(4)
            # ax.bar(x_space + 0.00 , marker_sides[:,0,0], width = 0.15)
            # ax.bar(x_space + 0.15 , marker_sides[:,0,1], width = 0.15)
            # ax.bar(x_space + 0.30 , marker_sides[:,0,2], width = 0.15)
            # ax.bar(x_space + 0.45 , marker_sides[:,0,3], width = 0.15)
            plt.bar(x_space + 0.00 , line_deltas[:], width = 0.15, color = (0.5,0.5,0.5,0.5))
            
            
            plt.pause(0.05)
            

    cv2.imshow("Frame",frame)
        
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows
vs.stop()