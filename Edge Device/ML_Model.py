# Import required libraries
import paho.mqtt.client as mqtt
import sys
import json
import base64
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
import time

# Setup the local client
LOCAL_MQTT_HOST = "localhost"
LOCAL_MQTT_PORT = 1883
LOCAL_MQTT_TOPIC_FROM_CAMERA = "to-model"
global interpreted_sign  #Empty string to store interpreted sign value
interpreted_sign = "~"
start_interpretation = False
interpretation_time = False
data_to_model = pd.DataFrame() # Empty dataframe to hold all the data to be sent to model
ROWS_PER_FRAME = 543 # Define number of rows per frame for model interpretation
data_collect = False
# Replace path to tflite model below
interpreter = tf.lite.Interpreter("/home/dpakapd/Documents/Berkeley/W251 Deep Learning in Edge/Final Project/Edge Device/Edge Device/v1_model.tflite")

# Add the decode_frame function
def decode_frame(encoded_frame):
    frame_data = base64.b64decode(encoded_frame)
    frame_np = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv.imdecode(frame_np, flags=1)
    return frame

def interpret_sign(data_df):
    global interpreter
    #Process dataframe sent by the program
    data_df.to_csv("landmarks.csv") #Overwritten for each frame
    data_df.to_parquet("landmarks.parquet")
    data_columns = ['x', 'y']
    model_data = data_df[['x','y']]
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
    #model_output = "Sign from Model"
    interpreted_sign = model_output
    return interpreted_sign

# Subscribe to local client topic
def on_connect_local(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    client.subscribe(LOCAL_MQTT_TOPIC_FROM_CAMERA)

def fill_data(data_df, frame_num):
    face_data = data_df[data_df.type == 'face']
    left_hand_data = data_df[data_df.type == 'left_hand']
    pose_data = data_df[data_df.type == 'pose']
    right_hand_data = data_df[data_df.type == 'right_hand']
    if len(face_data) == 0:
        face_data = pd.DataFrame({'frame': [frame_num] * 468, 'row-id':[np.nan] * 468,'type':['face'] * 468,
                                        'landmark_index':[np.nan] * 468,
                                        'x':[np.nan] * 468,
                                        'y':[np.nan] * 468})
    
    if len(left_hand_data) == 0:
        left_hand_data = pd.DataFrame({'frame': [frame_num] * 21, 'row-id':[np.nan] * 21,'type':['left_hand'] * 21,
                                        'landmark_index':[np.nan] * 21,
                                        'x':[np.nan] * 21,
                                        'y':[np.nan] * 21})
    if len(right_hand_data) == 0:
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
    

def on_message(client, userdata, msg):
    global start_interpretation, interpretation_time, interpreted_sign, data_to_model, data_collect
    
    try:
        # Receive the payload from MQTT
        response = json.loads(msg.payload.decode("utf-8"))
        # Combine the data into a single list
        all_data = response['face_data'] + response['left_hand_data'] + response['pose_data']+ response['right_hand_data']
        
        data_df = pd.DataFrame(all_data)
        frame_num = data_df.iloc[-1]['frame']
        data_df = fill_data(data_df, frame_num)
        data_df.drop(columns=['row-id'],inplace=True)
        
        # If user hits space bar, then collect data to send to the interpret_sign function.
        if start_interpretation:
            interpreted_sign = 'Start a sign'
            rows_in_frame, _ = data_df.shape #Check if the frame has 543 rows
            if rows_in_frame < 543: # If number of rows is less than 543
                padding_rows = 543 - rows_in_frame
                last_frame_value = data_df.iloc[-1]['frame'] #Get last frame value
                # Use the same frame number throughout the dataframe. Replace X and Y with NaN values to pad the dataframe
                padding_df = pd.DataFrame({'frame': [last_frame_value] * padding_rows,
                                           'type': [np.nan] * padding_rows,
                                           'landmark_index': [np.nan] * padding_rows,
                                            'x': [np.nan] * padding_rows,
                                            'y': [np.nan] * padding_rows})
                data_df = pd.concat([data_df, padding_df], ignore_index=True)
            data_to_model = pd.concat([data_to_model,data_df])
        if not start_interpretation and data_collect:
            interpreted_sign = interpret_sign(data_to_model)
        
        # Decode the frames and show the video. 
        encoded_frame = response["image"]
        frame = decode_frame(encoded_frame)
        cv.putText(frame, interpreted_sign, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('Detected Landmarks', frame)
        key = cv.waitKey(1)
        if key == ord(' '): #Check if spacebar is pressed by user
            data_collect = False
            if not start_interpretation:
                print("Starting time")
                interpretation_time = time.time() #Start the time to record 0.5 seconds
        elif key == 27: # When user presses the ESC key
            start_interpretation = False
            data_collect = True
            print("Stopping feed")
        elif key == ord('q'): # If the user presses 'Q' to quit
            cv.destroyAllWindows()
            exit()
        if interpretation_time != False and time.time() - interpretation_time >=0.5:
            print("Starting feed")
            start_interpretation = True
            #interpreted_sign = "Starting Processing"
            interpretation_time = False #Reset interpretation time so that you aren't calling this loop each time.

        
    except Exception as e:
        print("Unexpected error:", str(e))

# Create instance of local and remote clients and connect using the remote and local client port and host
local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)

local_mqttclient.on_message = on_message

# go into a loop
local_mqttclient.loop_forever()
