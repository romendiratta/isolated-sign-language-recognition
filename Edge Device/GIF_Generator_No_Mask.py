# Import required libraries
import pandas as pd
import cv2 as cv
import mediapipe as mp
import time
import base64
import json
import tensorflow as tf
import numpy as np
import imageio #Create GIF


# Global variables


global interpreted_sign  #Empty string to store interpreted sign value
interpreted_sign = "~"
start_interpretation = False
interpretation_time = False
data_to_model = pd.DataFrame() # Empty dataframe to hold all the data to be sent to model
ROWS_PER_FRAME = 543 # Define number of rows per frame for model interpretation
data_collect = False
space_first = False #Set bool variable to control start-stop sequence
timestamps = [] # Empty list to store timestamps for filtering last 0.5 seconds of data
gif_data = [] #Empty list to store GIF
# Replace path to tflite model below
interpreter = tf.lite.Interpreter("/home/dpakapd/Documents/Berkeley/W251 Deep Learning in Edge/Final Project/Edge Device/Edge Device/v1_model.tflite")
sign_map = {"25": "blow", "232": "wait", "48": "cloud", "23": "bird", "164": "owie", "67": "duck", "143": "minemy", "134": "lips", "86": "flower", "220": "time", "231": "vacuum", "8": "apple", "180": "puzzle", "144": "mitten", "216": "there", "65": "dry", "195": "shirt", "165": "owl", "243": "yellow", "156": "not", "249": "zipper", "45": "clean", "47": "closet", "181": "quiet", "108": "have", "30": "brother", "49": "clown", "41": "cheek", "54": "cute", "207": "store", "196": "shoe", "235": "wet", "193": "see", "70": "empty", "74": "fall", "14": "balloon", "89": "frenchfries", "80": "finger", "190": "same", "52": "cry", "121": "hungry", "162": "orange", "142": "milk", "97": "go", "62": "drawer", "0": "TV", "6": "another", "93": "giraffe", "233": "wake", "19": "bee", "13": "bad", "35": "can", "191": "say", "34": "callonphone", "81": "finish", "159": "old", "12": "backyard", "198": "sick", "136": "look", "215": "that", "24": "black", "246": "yourself", "161": "open", "4": "alligator", "146": "moon", "78": "find", "172": "pizza", "194": "shhh", "76": "fast", "125": "jacket", "192": "scissors", "157": "now", "140": "man", "206": "sticky", "127": "jump", "199": "sleep", "210": "sun", "83": "first", "101": "grass", "228": "uncle", "84": "fish", "51": "cowboy", "203": "snow", "66": "dryer", "102": "green", "32": "bug", "150": "nap", "77": "feet", "247": "yucky", "147": "morning", "189": "sad", "73": "face", "169": "penny", "92": "gift", "152": "night", "104": "hair", "239": "who", "217": "think", "31": "brown", "138": "mad", "17": "bed", "63": "drink", "205": "stay", "85": "flag", "223": "tooth", "11": "awake", "214": "thankyou", "120": "hot", "132": "like", "237": "where", "115": "hesheit", "176": "potty", "61": "down", "209": "stuck", "153": "no", "110": "head", "87": "food", "178": "pretty", "158": "nuts", "5": "animal", "90": "frog", "21": "beside", "154": "noisy", "234": "water", "236": "weus", "105": "happy", "238": "white", "33": "bye", "117": "high", "79": "fine", "27": "boat", "3": "all", "219": "tiger", "168": "pencil", "200": "sleepy", "99": "grandma", "44": "chocolate", "109": "haveto", "182": "radio", "75": "farm", "7": "any", "248": "zebra", "183": "rain", "226": "toy", "60": "donkey", "133": "lion", "64": "drop", "141": "many", "15": "bath", "10": "aunt", "241": "will", "107": "hate", "160": "on", "177": "pretend", "129": "kitty", "82": "fireman", "20": "before", "59": "doll", "204": "stairs", "128": "kiss", "137": "loud", "114": "hen", "135": "listen", "95": "give", "242": "wolf", "55": "dad", "103": "gum", "111": "hear", "186": "refrigerator", "163": "outside", "53": "cut", "229": "underwear", "173": "please", "42": "child", "201": "smile", "167": "pen", "245": "yesterday", "119": "horse", "171": "pig", "211": "table", "72": "eye", "202": "snack", "208": "story", "174": "police", "9": "arm", "212": "talk", "100": "grandpa", "222": "tongue", "175": "pool", "94": "girl", "230": "up", "22": "better", "227": "tree", "56": "dance", "46": "close", "213": "taste", "43": "chin", "187": "ride", "16": "because", "123": "if", "38": "cat", "240": "why", "37": "carrot", "58": "dog", "148": "mouse", "126": "jeans", "197": "shower", "131": "later", "145": "mom", "155": "nose", "244": "yes", "2": "airplane", "28": "book", "26": "blue", "122": "icecream", "91": "garbage", "221": "tomorrow", "185": "red", "50": "cow", "170": "person", "179": "puppy", "39": "cereal", "225": "touch", "149": "mouth", "29": "boy", "218": "thirsty", "139": "make", "88": "for", "96": "glasswindow", "124": "into", "184": "read", "71": "every", "18": "bedroom", "151": "napkin", "68": "ear", "224": "toothbrush", "118": "home", "166": "pajamas", "113": "hello", "112": "helicopter", "130": "lamp", "188": "room", "57": "dirty", "40": "chair", "106": "hat", "69": "elephant", "1": "after", "36": "car", "116": "hide", "98": "goose"}


def fill_data(data_df, frame_num):
    face_data = data_df[data_df['type'] == 'face']
    left_hand_data = data_df[data_df['type'] == 'left_hand']
    pose_data = data_df[data_df['type'] == 'pose']
    right_hand_data = data_df[data_df['type'] == 'right_hand']
    if len(face_data) == 0:
        face_data = pd.DataFrame({'frame': [frame_num] * 468, 'row-id':[np.nan] * 468,'type':['face'] * 468,
                                        'landmark_index':[np.nan] * 468,
                                        'x':[np.nan] * 468,
                                        'y':[np.nan] * 468})
    
    if len(left_hand_data) == 0:
        #print("Inside left hand")
        left_hand_data = pd.DataFrame({'frame': [frame_num] * 21, 'row-id':[np.nan] * 21,'type':['left_hand'] * 21,
                                        'landmark_index':[np.nan] * 21,
                                        'x':[np.nan] * 21,
                                        'y':[np.nan] * 21})
    if len(right_hand_data) == 0:
        #print("Inside right hand")
        right_hand_data = pd.DataFrame({'frame': [frame_num] * 21, 'row-id':[np.nan] * 21,'type':['right_hand'] * 21,
                                        'landmark_index':[np.nan] * 21,
                                        'x':[np.nan] * 21,
                                        'y':[np.nan] * 21})
    if len(pose_data) == 0:
        pose_data = pd.DataFrame({'frame': [frame_num] * 33, 'row-id':[np.nan] * 33,'type':['pose'] * 33,
                                        'landmark_index':[np.nan] * 33,
                                        'x':[np.nan] * 33,
                                        'y':[np.nan] * 33})
        
    all_data = pd.concat([face_data,left_hand_data,pose_data,right_hand_data])
    return all_data


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
    #face_data = pd.DataFrame(face_data)
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
    #pose_data = pd.DataFrame(pose_data)
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
    #left_hand_data = pd.DataFrame(left_hand_data)
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
    #right_hand_data = pd.DataFrame(right_hand_data)
    return right_hand_data

# Function to convert the video frames into byte array for streaming
def encode_frame(frame):
    _, buffer = cv.imencode('.jpg', frame)
    frame_encoded = base64.b64encode(buffer)
    return frame_encoded.decode('utf-8')


# Let's stream image from the camera and save it
cap = cv.VideoCapture(0) #0 is listed here as it is the camera on my computer. Change it as needed to yours

frame_number = 0
#face_data = pd.DataFrame()
#left_hand_data = pd.DataFrame()
#pose_data = pd.DataFrame()
#right_hand_data = pd.DataFrame()
start = time.time()
with face_mesh.FaceMesh(static_image_mode = False, max_num_faces = max_faces, min_detection_confidence=0.5, min_tracking_confidence= 0.5) as your_face, mp_pose.Pose(min_detection_confidence=0.5) as your_pose, mp_hands.Hands(min_detection_confidence=0.5) as your_hands:
    while True: #time.time() - start <2: 
        face_data = []
        left_hand_data = []
        pose_data = []
        right_hand_data = []
        ret, image = cap.read() # Capture a frame from the webcam
        if not ret: # If image not detected, then break out
            break

        image = cv.cvtColor(image,cv.COLOR_BGR2RGB) # Convert image to RBG as Mediapipe uses RGB
        image.flags.writeable = False
        faces = your_face.process(image) # Detect faces in captured image
        image.flags.writeable = True
        if faces.multi_face_landmarks: # If face is detected
            for face_landmarks in faces.multi_face_landmarks: # For each landmark detected in the face
                #landmark_draw.draw_landmarks(image, face_landmarks, landmark_holistic.FACEMESH_CONTOURS,
                #                             landmark_draw.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1),
                #                            landmark_draw.DrawingSpec(color=(0,255,255),thickness=1,circle_radius=1)) #Draw the landmark on the captured image
                face_data = extract_face_landmarks(face_landmarks, frame_number)
                
        poses = your_pose.process(image) #Process image to capture the pose
        if poses.pose_landmarks: # Check if poses are detected
            # Draw landmarks on detected pose
            #landmark_draw.draw_landmarks(image, poses.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            pose_data = extract_pose_landmarks(poses.pose_landmarks, frame_number)
            
        hands = your_hands.process(image) #Let's do the right hand first
        if hands.multi_hand_landmarks: # If hands are detected
            for hand_landmarks, handedness in zip(hands.multi_hand_landmarks or [], hands.multi_handedness or []):
                hand_label = handedness.classification[0].label
                if hand_label == "Left":
                    left_hand_data = extract_left_hand_data(hand_landmarks, frame_number)
                    #landmark_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                elif hand_label == "Right":
                    right_hand_data = extract_right_hand_data(hand_landmarks, frame_number)
                    #landmark_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        image = cv.cvtColor(image,cv.COLOR_RGB2BGR) # Convert image back to normal from RGB
        gif_image = cv.cvtColor(image,cv.COLOR_RGB2BGR) # Convert image back to normal from RGB
        #data_df = pd.concat([face_data, left_hand_data, pose_data, right_hand_data], ignore_index=True)
        # If user hits space bar, then collect data to send to the interpret_sign function.
        if start_interpretation and not data_collect:
            all_data = face_data + right_hand_data + pose_data + left_hand_data
            data_df = pd.DataFrame(all_data)
            data_df = fill_data(data_df, frame_number)
            data_to_model = pd.concat([data_to_model, data_df])
            data_to_model.drop(columns=['row-id'],inplace=True)
            interpreted_sign = 'Start a sign'
            timestamps.append(time.time()) # Collect timestamp for each iteration
            data_to_model = pd.concat([data_to_model,data_df])
        if not start_interpretation and data_collect:
            #interpreted_sign = interpret_sign(data_to_model, frame_number)
            time_data = pd.Series(timestamps) # Convert timestamp into a series
            data_to_model['timestamp'] = time_data[time_data<(time.time() - 0.5)] # Drop the last 0.5 seconds of data
            timestamps = [] #Re-initialize the list to empty
            data_columns = ['x','y']
            try:
                model_data = data_to_model[['x','y']]
                n_frames = int(len(model_data) / ROWS_PER_FRAME)
                model_data = model_data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
                model_data =  model_data.astype(np.float32)
                found_signatures = list(interpreter.get_signature_list().keys())
                prediction_fn = interpreter.get_signature_runner("serving_default")

                output = prediction_fn(inputs=model_data)
                if output['outputs'].max() < 0.001:
                    model_output = "Not detected"
                else:
                    model_output = str(output['outputs'].argmax())
                    model_confidence = str(round(output['outputs'].max() * 100,2))
                if model_output == "0":
                    model_output = "No sign detected. Please try again"
                elif output['outputs'].max() < 0.4:
                    model_output = sign_map[model_output]
                    model_output = model_output + " - " + model_confidence + "%: Low Confidence"
                else:
                    model_output = sign_map[model_output]
                    model_output = model_output + " - " + model_confidence + "%"
                #model_output = "Sign from Model"
                interpreted_sign = model_output
                data_to_model = pd.DataFrame() #Reset the dataframe
            except:
                pass
        #print(frame_number)

        cv.putText(gif_image, interpreted_sign, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(image, interpreted_sign, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('Detected Landmarks', image)
        gif_data.append(gif_image) # Append the image to create GIF
        key = cv.waitKey(1)
        if key == ord(' ') and not space_first: #Check if spacebar is pressed by user
            data_to_model = pd.DataFrame()
            data_collect = False
            if not start_interpretation:
                print("Starting time")
                space_first = True
                interpretation_time = time.time() #Start the time to record 0.5 seconds
        #elif key == 27: # When user presses the ESC key
        elif key == ord(' ') and space_first:
            if start_interpretation:
                start_interpretation = False
                space_first = False
                data_collect = True
                print("Stopping feed")
        elif key == ord('q'): # If the user presses 'Q' to quit
            cv.destroyAllWindows()
            print("Saving to GIF")
            imageio.mimsave("presentation.gif", gif_data, fps = 20)
            exit()
        if interpretation_time != False and time.time() - interpretation_time >=0.5:
            print("Starting feed")
            start_interpretation = True
            #interpreted_sign = "Starting Processing"
            interpretation_time = False #Reset interpretation time so that you aren't calling this loop each time.
        
        frame_number += 1
        

        if cv.waitKey(1) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()