import paho.mqtt.client as mqtt
import sys

LOCAL_HOST = 'localhost'
LOCAL_PORT = 1883
LOCAL_TOPIC = 'test-topic'

# Subscribe to local client topic
def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(LOCAL_TOPIC)


def on_message(client,userdata, msg):
  try:
    # if we wanted to re-publish this message, something like this should work
    #print("Sending payload")
    msg = msg.payload #Receive the payload sent by cam.py (publisher)
    print(str(msg))
  except:
    print("Unexpected error:", sys.exc_info()[0])

#Create instance of local and remote clients and connect using the remote and local client port and host
local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_HOST, LOCAL_PORT, 60)
local_mqttclient.on_message = on_message

# go into a loop
local_mqttclient.loop_forever()