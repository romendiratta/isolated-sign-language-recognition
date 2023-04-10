# Import required libraries
import pandas as pd
import cv2 as cv
import mediapipe as mp
import time
import base64
import paho.mqtt.client as mqtt
import json


# MQTT settings
LOCAL_MQTT_HOST = "localhost"
LOCAL_MQTT_PORT = 1883
LOCAL_MQTT_TOPIC_TO_MODEL = "to-model"


# Initialize MQTT client
mqttclient = mqtt.Client()
mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
mqttclient.loop_start()

# Let's initialize models for face, pose and hands
face_mesh = mp.solutions.face_mesh # This is to draw the mesh on the face where each point is detected
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Max num of faces to detect
max_faces = 1

# Mediapipe uses drawing_utils function to draw landmark on images captured using OpenCV. Hence we need to initialize that
landmark_draw = mp.solutions.drawing_utils
landmark_styles = mp.solutions.drawing_styles
landmark_holistic = mp.solutions.holistic

# Since we have to draw a mesh, let's set thickness and radius of dots in the face
drawing_spec = landmark_draw.DrawingSpec(thickness=1, circle_radius=1)

t_end = time.time() + 25 # Let's stream the video only for 25 seconds.

# Let's define functions to extract the landmarks for face, pose and both hands
def extract_face_landmarks(landmarks, frame_num):
    data_type = 'face'
    face_data = []
    for idx, landmark in enumerate(landmarks.landmark):
        row_id = str(frame_num) + "-"+data_type+"-"+str(idx)
        face_data.append({
            'frame':frame_num,
            'row-id':row_id,
            'type':data_type,
            'landmark_index': idx,
            'x': landmark.x,
            'y': landmark.y,
        })
    return face_data

def extract_pose_landmarks(landmarks, frame_num):
    data_type = 'pose'
    pose_data = []
    for idx, landmark in enumerate(landmarks.landmark):
        row_id = str(frame_num) + "-"+data_type+"-"+str(idx)
        pose_data.append({
            'frame':frame_num,
            'row-id':row_id,
            'type':data_type,
            'landmark_index': idx,
            'x': landmark.x,
            'y': landmark.y,
        })
    return pose_data

def extract_left_hand_data(landmarks, frame_num):
    data_type = 'left_hand'
    left_hand_data = []
    for idx, landmark in enumerate(landmarks.landmark):
        row_id = str(frame_num) + "-"+data_type+"-"+str(idx)
        left_hand_data.append({
            'frame':frame_num,
            'row-id':row_id,
            'type':data_type,
            'landmark_index': idx,
            'x': landmark.x,
            'y': landmark.y,
        })
    return left_hand_data

def extract_right_hand_data(landmarks, frame_num):
    data_type = 'right_hand'
    right_hand_data = []
    for idx, landmark in enumerate(landmarks.landmark):
        row_id = str(frame_num) + "-"+data_type+"-"+str(idx)
        right_hand_data.append({
            'frame':frame_num,
            'row-id':row_id,
            'type':data_type,
            'landmark_index': idx,
            'x': landmark.x,
            'y': landmark.y,
        })
    return right_hand_data

# Function to convert the video frames into byte array for streaming
def encode_frame(frame):
    _, buffer = cv.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer)
    return frame_encoded.decode('utf-8')


# Let's stream image from the camera and save it
cap = cv.VideoCapture(0) #0 is listed here as it is the camera on my computer. Change it as needed to yours

frame_number = 0

message = 'Processing...'
with face_mesh.FaceMesh(static_image_mode = False, max_num_faces = max_faces, min_detection_confidence=0.5, min_tracking_confidence= 0.5) as your_face, mp_pose.Pose(min_detection_confidence=0.5) as your_pose, mp_hands.Hands(min_detection_confidence=0.5) as your_hands:
    while True: #time.time() < t_end: # While time < 25 seconds
        face_data = []
        right_hand_data = [] #Declare empty list so that it can be used if no hands or pose is detected
        left_hand_data = []
        pose_data = []
        ret, image = cap.read() # Capture a frame from the webcam
        if not ret: # If image not detected, then break out
            break

        image = cv.cvtColor(image,cv.COLOR_BGR2RGB) # Convert image to RBG as Mediapipe uses RGB
        image.flags.writeable = False
        faces = your_face.process(image) # Detect faces in captured image
        image.flags.writeable = True
        if faces.multi_face_landmarks: # If face is detected
            for face_landmarks in faces.multi_face_landmarks: # For each landmark detected in the face
                landmark_draw.draw_landmarks(image, face_landmarks, landmark_holistic.FACEMESH_CONTOURS,
                                             landmark_draw.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1),
                                            landmark_draw.DrawingSpec(color=(0,255,255),thickness=1,circle_radius=1)) #Draw the landmark on the captured image
                face_data = extract_face_landmarks(face_landmarks, frame_number)

        poses = your_pose.process(image) #Process image to capture the pose
        if poses.pose_landmarks: # Check if poses are detected
            # Draw landmarks on detected pose
            landmark_draw.draw_landmarks(image, poses.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            pose_data = extract_pose_landmarks(poses.pose_landmarks, frame_number)
        
        hands = your_hands.process(image) #Let's do the right hand first
        if hands.multi_hand_landmarks: # If hands are detected
            for hand_landmarks, handedness in zip(hands.multi_hand_landmarks or [], hands.multi_handedness or []):
                hand_label = handedness.classification[0].label
                if hand_label == "Left":
                    left_hand_data = extract_left_hand_data(hand_landmarks, frame_number)
                    landmark_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                elif hand_label == "Right":
                    right_hand_data = extract_right_hand_data(hand_landmarks, frame_number)
                    landmark_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        image = cv.cvtColor(image,cv.COLOR_RGB2BGR) # Convert image back to normal from RGB

        encoded_frame = encode_frame(image)

        # Send data to model.py
        data_to_model = {
            'face_data': face_data,
            'pose_data': pose_data,
            'left_hand_data': left_hand_data,
            'right_hand_data': right_hand_data,
            'image':encoded_frame
        }
        print(len(face_data), len(left_hand_data), len(pose_data), len(right_hand_data))
        
        mqttclient.publish(LOCAL_MQTT_TOPIC_TO_MODEL, json.dumps(data_to_model))
        frame_number += 1
        

        if cv.waitKey(1) == ord('q'):
            break

cap.release()
mqttclient.loop_stop()
cv.destroyAllWindows()