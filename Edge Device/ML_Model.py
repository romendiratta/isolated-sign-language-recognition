import paho.mqtt.client as mqtt
import sys
import json
import base64
import cv2 as cv
import numpy as np

# Setup the local client
LOCAL_MQTT_HOST = "localhost"
LOCAL_MQTT_PORT = 1883
LOCAL_MQTT_TOPIC_FROM_CAMERA = "to-model"


# Add the decode_frame function
def decode_frame(encoded_frame):
    frame_data = base64.b64decode(encoded_frame)
    frame_np = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv.imdecode(frame_np, flags=1)
    return frame

# Subscribe to local client topic
def on_connect_local(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    client.subscribe(LOCAL_MQTT_TOPIC_FROM_CAMERA)

def on_message(client, userdata, msg):
    try:
        message = "Processing at Model..."
        print("Receiving payload")
        response = json.loads(msg.payload)

        encoded_frame = response["image"]
        frame = decode_frame(encoded_frame)
        cv.putText(frame, message, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('Detected Landmarks', frame)
        cv.waitKey(1)
        
    except Exception as e:
        print("Unexpected error:", str(e))

# Create instance of local and remote clients and connect using the remote and local client port and host
local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)

local_mqttclient.on_message = on_message

# go into a loop
local_mqttclient.loop_forever()
