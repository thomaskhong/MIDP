from turtle import shape
import cv2
import mediapipe as mp
import numpy as np
import math

## --- Function definitions --- ###

# calc_line_length calculates the distance between two points in the form of (x1,y1) and (x2,y2)
def calc_line_length(point1,point2):
    line_length = math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    return line_length

# calc_finger_length takes as input:
#   - finger index (ranging 0 to 4 and equivalent to thumb to pinkie)
#   - landmark indexes over which to add up the finger length (these are based on how mediapipe labels the individual joints)
#   - landmark coordinates (an array of the x and y coordinates of each joint)
# The function then iterates through each segment of the finger (i.e.: between each joint) and adds up the distance between them
def calc_finger_length(finger_index, landmark_sum_indexes, hand_landmark_coordinates):
    index_list = landmark_sum_indexes[finger_index]
    index_list = np.arange(index_list[0],index_list[1],-1)
    finger_length = 0
    for index in index_list:
        segment_length = calc_line_length(hand_landmark_coordinates[index][0],hand_landmark_coordinates[index-1][0])
        finger_length = finger_length + segment_length
    return finger_length


## --- Initialisation of parameters --- ###

# Init of MediaPipe objects
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Init of Aruco objects
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
arucoParams = cv2.aruco.DetectorParameters_create()

# Init of calculation and storage data arrays
marker_length_averages = np.zeros(4) # 4 values that store the average marker length for each marker
hand_landmark_coordinates = np.zeros((21,1,2)) # 21, 1-by-2 arrays that store the x and y position coordinates of each joint
finger_lengths = np.zeros(5) # 5 values that store the length of the thumb to pinkie
landmark_sum_indexes = [[4,2],[8,5],[12,9],[16,13],[20,17]] # array of which points correspond to each finger

# Init of txt document object to which data will be written to
doc = open('C:/Users/khong/OneDrive/Documents/MIDP/Grasshopper/data_output.txt', 'w')

## --- Initialisation of parameters --- ###

# Start the video capture thread. Change the argument of VideoCapture() to change the webcam from which the feed is taken.
cap = cv2.VideoCapture(1)

# Main loop
while True:
    # Read a single frame from the video capture thread.
    success, image = cap.read()
    # Inform user if the frame is empty
    if not success:
            print("Ignoring empty camera frame")
            continue

    # Detect the Aruco markers from the video feed
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    # Continue with calculations only if at least one Aruco code is detected
    # Also ensures that the calculations are only performed if the right Aruco codes are detected (i.e.: the 0th to 3rd codes). The detection would sometimes mistake one of them for a different code and try to use that as an index, throwing an out of bounds error 
    if len(corners) > 0:
        # flattens the ids array so that it can be more easily accessed
        ids = ids.flatten()

        if np.all(ids<4):

            # initialise a fresh storage for measurements of:
            marker_sides = np.zeros((4,1,4)) # each side length of the marker for each marker
            line_deltas = np.zeros((4)) # the difference between the average of the horizontal marker sides and the average of the vertical marker sides for each marker

            # looping through each detected marker
            for (markerCorner, markerID) in zip(corners, ids):
                # extract each individual corner from an Aruco marker
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                # parse the corners into individual variables
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # calculate the distance between each corner, thus giving the length of the edges of the marker
                topLine_length = int(calc_line_length(topLeft,topRight))
                rightLine_length = int(calc_line_length(topRight,bottomRight))
                bottomLine_length = int(calc_line_length(bottomLeft,bottomRight))
                leftLine_length = int(calc_line_length(topLeft,bottomLeft))

                # put the edge lengths into appropriate location in the marker_sides variable
                marker_sides[markerID] = (topLine_length,rightLine_length,bottomLine_length,leftLine_length)

                # calculate the horizontal and vertical averages of the marker lengths, then place these in storage
                line_averages = ((topLine_length+bottomLine_length)/2 , (leftLine_length+rightLine_length)/2)
                marker_length_averages[markerID] = np.mean(line_averages)

                # calculate the absolute difference between the horizontal and vertical averages
                line_deltas[markerID] = math.sqrt((line_averages[0]-line_averages[1])**2)

                # the image is experiencing a large amount of perspective distortion if the difference between the horizontal and vertical averages is larger than a percentage of the average length of the marker edges (set arbitrarily at 5% ~ 2 pixels for a relatively low capture resolution)
                # if perspective distortion is present in the currently considered Aruco marker, colour it red. Otherwise colour it green.
                if np.any(line_deltas > max(line_averages)*0.05):
                    g = 0
                    r = 255
                else:
                    r = 0
                    g = 255
                
                # create the lines between each marker corner
                cv2.line(image, topLeft, topRight, (0, g, r), 2)
                cv2.line(image, topRight, bottomRight, (0, g, r), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, g, r), 2)
                cv2.line(image, bottomLeft, topLeft, (0, g, r), 2)

                # compute and draw the center (x, y)-coordinates of the ArUco marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

                # draw the ArUco marker ID on the frame
                cv2.putText(image, str(markerID),
                (topLeft[0], topLeft[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

            # start using the MediaPipe hand model with the chosen detection settings
            with mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
                
                # set captured frame as read-only for performance
                image.flags.writeable = False
                # get the height and width of the captured frame to normalise the pixel coordinates later
                imageHeight, imageWidth, _ = image.shape
                # convert captured frame into RGB as cv2 captures in BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # detect hands in the captured frame
                results = hands.process(image)

                # perform calculations only if hand landmarks are detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # for each hand landmark, draw the landmark and the connections onto the frame
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                        # for each hand landmark, extract the normalized landmark (ranges from 0 to 1), then calculate the pixel coordinates and store
                        for point_num,point in enumerate(mp_hands.HandLandmark):
                            normalizedLandmark = hand_landmarks.landmark[point]
                            pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
                            hand_landmark_coordinates[point_num] = np.array(pixelCoordinatesLandmark)

            # loop through each finger and calculate length
            for finger_num in range(0,5):
                finger_lengths[finger_num] = round((calc_finger_length(finger_num, landmark_sum_indexes, hand_landmark_coordinates)*31)/np.mean(marker_length_averages),0)
                # print(finger_lengths[finger_num])
            print(finger_lengths)
            
    # show the frame with all information rendered on it
    cv2.imshow("Frame",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # define stop key and program break
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        # write all the measured data to a txt document
        for i,x in enumerate(finger_lengths):
            doc.write(str(x) + '\n')
        break

# terminate the video capture thread
cap.release()

# close txt file to which data was written
doc.close()