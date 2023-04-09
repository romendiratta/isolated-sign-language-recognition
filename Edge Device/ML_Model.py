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


# Add the decode_frame function
def decode_frame(encoded_frame):
    frame_data = base64.b64decode(encoded_frame)
    frame_np = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv.imdecode(frame_np, flags=1)
    return frame

def interpret_sign(data_df):
    #Process dataframe sent by the program
    print(data_df.shape)
    model_output = "Sign from Model"
    interpreted_sign = model_output    

# Subscribe to local client topic
def on_connect_local(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    client.subscribe(LOCAL_MQTT_TOPIC_FROM_CAMERA)

def on_message(client, userdata, msg):
    global start_interpretation, interpretation_time
    try:
        # Receive the payload from MQTT
        response = json.loads(msg.payload.decode("utf-8"))
        # Combine the data into a single list
        all_data = response['face_data'] + response['pose_data'] + response['left_hand_data'] + response['right_hand_data']
        # Convert the list to a DataFrame
        data_df = pd.DataFrame(all_data)
        # If user hits space bar, then send data to the interpret_sign function.
        if start_interpretation:
            interpret_sign(data_df)
        
        # Decode the frames and show the video. 
        encoded_frame = response["image"]
        frame = decode_frame(encoded_frame)
        cv.putText(frame, interpreted_sign, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('Detected Landmarks', frame)
        key = cv.waitKey(1)
        if key == ord(' '): #Check if spacebar is pressed by user
            if not start_interpretation:
                interpretation_time = time.time() #Start the time to record 0.5 seconds
        elif key == 27: # When user presses the ESC key
            start_interpretation = False
            print("Stopping feed")
        elif key == ord('q'): # If the user presses 'Q' to quit
            cv.destroyAllWindows()
            exit()
        if interpretation_time != False and time.time() - interpretation_time >=0.5:
            print("Starting feed")
            start_interpretation = True
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
