# Import required libraries
import paho.mqtt.client as mqtt
import sys
import json
import base64
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf

# Setup the local client
LOCAL_MQTT_HOST = "localhost"
LOCAL_MQTT_PORT = 1883
LOCAL_MQTT_TOPIC_FROM_CAMERA = "to-model"
global interpreted_sign  #Empty string to store interpreted sign value
interpreted_sign = "~"


# Add the decode_frame function
def decode_frame(encoded_frame):
    frame_data = base64.b64decode(encoded_frame)
    frame_np = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv.imdecode(frame_np, flags=1)
    return frame

def interpret_sign(data_df):
    #Process dataframe sent by the program
    print(data_df.shape)
    data_df.to_csv("landmarks.csv")

    model_output = "Sign from Model"
    interpreted_sign = model_output    

# Subscribe to local client topic
def on_connect_local(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    client.subscribe(LOCAL_MQTT_TOPIC_FROM_CAMERA)

def on_message(client, userdata, msg):
    try:
        # Receive the payload from MQTT
        response = json.loads(msg.payload.decode("utf-8"))
        # Combine the data into a single list
        all_data = response['face_data'] + response['pose_data'] + response['left_hand_data'] + response['right_hand_data']
        # Convert the list to a DataFrame
        data_df = pd.DataFrame(all_data)        
        interpret_sign(data_df)
        encoded_frame = response["image"]
        frame = decode_frame(encoded_frame)
        cv.putText(frame, interpreted_sign, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('Detected Landmarks', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            cv.destroyAllWindows()
            exit()
        
    except Exception as e:
        print("Unexpected error:", str(e))

# Create instance of local and remote clients and connect using the remote and local client port and host
local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)

local_mqttclient.on_message = on_message

# go into a loop
local_mqttclient.loop_forever()
