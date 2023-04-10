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
sign_map = {"25": "blow", "232": "wait", "48": "cloud", "23": "bird", "164": "owie", "67": "duck", "143": "minemy", "134": "lips", "86": "flower", "220": "time", "231": "vacuum", "8": "apple", "180": "puzzle", "144": "mitten", "216": "there", "65": "dry", "195": "shirt", "165": "owl", "243": "yellow", "156": "not", "249": "zipper", "45": "clean", "47": "closet", "181": "quiet", "108": "have", "30": "brother", "49": "clown", "41": "cheek", "54": "cute", "207": "store", "196": "shoe", "235": "wet", "193": "see", "70": "empty", "74": "fall", "14": "balloon", "89": "frenchfries", "80": "finger", "190": "same", "52": "cry", "121": "hungry", "162": "orange", "142": "milk", "97": "go", "62": "drawer", "0": "TV", "6": "another", "93": "giraffe", "233": "wake", "19": "bee", "13": "bad", "35": "can", "191": "say", "34": "callonphone", "81": "finish", "159": "old", "12": "backyard", "198": "sick", "136": "look", "215": "that", "24": "black", "246": "yourself", "161": "open", "4": "alligator", "146": "moon", "78": "find", "172": "pizza", "194": "shhh", "76": "fast", "125": "jacket", "192": "scissors", "157": "now", "140": "man", "206": "sticky", "127": "jump", "199": "sleep", "210": "sun", "83": "first", "101": "grass", "228": "uncle", "84": "fish", "51": "cowboy", "203": "snow", "66": "dryer", "102": "green", "32": "bug", "150": "nap", "77": "feet", "247": "yucky", "147": "morning", "189": "sad", "73": "face", "169": "penny", "92": "gift", "152": "night", "104": "hair", "239": "who", "217": "think", "31": "brown", "138": "mad", "17": "bed", "63": "drink", "205": "stay", "85": "flag", "223": "tooth", "11": "awake", "214": "thankyou", "120": "hot", "132": "like", "237": "where", "115": "hesheit", "176": "potty", "61": "down", "209": "stuck", "153": "no", "110": "head", "87": "food", "178": "pretty", "158": "nuts", "5": "animal", "90": "frog", "21": "beside", "154": "noisy", "234": "water", "236": "weus", "105": "happy", "238": "white", "33": "bye", "117": "high", "79": "fine", "27": "boat", "3": "all", "219": "tiger", "168": "pencil", "200": "sleepy", "99": "grandma", "44": "chocolate", "109": "haveto", "182": "radio", "75": "farm", "7": "any", "248": "zebra", "183": "rain", "226": "toy", "60": "donkey", "133": "lion", "64": "drop", "141": "many", "15": "bath", "10": "aunt", "241": "will", "107": "hate", "160": "on", "177": "pretend", "129": "kitty", "82": "fireman", "20": "before", "59": "doll", "204": "stairs", "128": "kiss", "137": "loud", "114": "hen", "135": "listen", "95": "give", "242": "wolf", "55": "dad", "103": "gum", "111": "hear", "186": "refrigerator", "163": "outside", "53": "cut", "229": "underwear", "173": "please", "42": "child", "201": "smile", "167": "pen", "245": "yesterday", "119": "horse", "171": "pig", "211": "table", "72": "eye", "202": "snack", "208": "story", "174": "police", "9": "arm", "212": "talk", "100": "grandpa", "222": "tongue", "175": "pool", "94": "girl", "230": "up", "22": "better", "227": "tree", "56": "dance", "46": "close", "213": "taste", "43": "chin", "187": "ride", "16": "because", "123": "if", "38": "cat", "240": "why", "37": "carrot", "58": "dog", "148": "mouse", "126": "jeans", "197": "shower", "131": "later", "145": "mom", "155": "nose", "244": "yes", "2": "airplane", "28": "book", "26": "blue", "122": "icecream", "91": "garbage", "221": "tomorrow", "185": "red", "50": "cow", "170": "person", "179": "puppy", "39": "cereal", "225": "touch", "149": "mouth", "29": "boy", "218": "thirsty", "139": "make", "88": "for", "96": "glasswindow", "124": "into", "184": "read", "71": "every", "18": "bedroom", "151": "napkin", "68": "ear", "224": "toothbrush", "118": "home", "166": "pajamas", "113": "hello", "112": "helicopter", "130": "lamp", "188": "room", "57": "dirty", "40": "chair", "106": "hat", "69": "elephant", "1": "after", "36": "car", "116": "hide", "98": "goose"}

# Add the decode_frame function
def decode_frame(encoded_frame):
    frame_data = base64.b64decode(encoded_frame)
    frame_np = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv.imdecode(frame_np, flags=1)
    return frame

def interpret_sign(data_df):
    global interpreter, sign_map
    #Process dataframe sent by the program
    #temp_df = pd.read_csv("landmarks.csv")
    #temp_df = pd.concat([temp_df,data_df])
    #temp_df.to_parquet("landmarks.parquet")
    #temp_df.to_csv("landmarks.csv") 
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
    if model_output == "0":
        model_output = "No sign detected. Please try again"
    else:
        model_output = sign_map[model_output]    
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
    

def on_message(client, userdata, msg):
    global start_interpretation, interpretation_time, interpreted_sign, data_to_model, data_collect
    
    try:
        # Receive the payload from MQTT
        response = json.loads(msg.payload.decode("utf-8"))
        #print(len(response['face_data']), len(response['left_hand_data']), len(response['pose_data']), len(response['right_hand_data']))
        # If user hits space bar, then collect data to send to the interpret_sign function.
        if start_interpretation and not data_collect:
            # Combine the data into a single list
            all_data = response['face_data'] + response['left_hand_data'] + response['pose_data']+ response['right_hand_data']
            data_df = pd.DataFrame(all_data)
            frame_num = data_df.iloc[-1]['frame']
            data_df = fill_data(data_df, frame_num)
            data_df.drop(columns=['row-id'],inplace=True)
            interpreted_sign = 'Start a sign'
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
            data_to_model = pd.DataFrame()
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
