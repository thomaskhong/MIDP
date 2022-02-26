from turtle import shape
import cv2
import mediapipe as mp
import numpy as np
import math

def calc_line_length(point1,point2):
    line_length = math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    return line_length

def calc_finger_length(finger_index, landmark_sum_indexes, hand_landmark_coordinates):
    index_list = landmark_sum_indexes[finger_index]
    index_list = np.arange(index_list[0],index_list[1],-1)
    finger_length = 0
    for index in index_list:
        segment_length = calc_line_length(hand_landmark_coordinates[index][0],hand_landmark_coordinates[index-1][0])
        finger_length = finger_length + segment_length
    return finger_length

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hand_landmark_coordinates = np.zeros((21,1,2))
finger_lengths = np.zeros(5)
landmark_sum_indexes = [[4,2],[8,5],[12,9],[16,13],[20,17]]

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue

        image.flags.writeable = False
        imageHeight, imageWidth, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for point_num,point in enumerate(mp_hands.HandLandmark):
                    normalizedLandmark = hand_landmarks.landmark[point]
                    pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
                    hand_landmark_coordinates[point_num] = np.array(pixelCoordinatesLandmark)

        cv2.imshow('MediaPipe Hands', image)

        test = calc_finger_length(0, landmark_sum_indexes, hand_landmark_coordinates)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cap.release()