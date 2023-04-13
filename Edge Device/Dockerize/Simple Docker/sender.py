import paho.mqtt.client as mqtt
import sys

LOCAL_HOST = 'localhost'
LOCAL_PORT = 1883
LOCAL_TOPIC = 'test-topic'

# Subscribe to local client topic
def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))


#Create instance of local and remote clients and connect using the remote and local client port and host
local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_HOST, LOCAL_PORT, 60)
local_mqttclient.loop_start()

while True:
        message = "Received the message at receiver"
        local_mqttclient.publish(LOCAL_TOPIC,message)

# go into a loop
local_mqttclient.loop_forever()