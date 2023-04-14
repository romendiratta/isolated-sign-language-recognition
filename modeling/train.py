import sys
import os
import json
import wandb
import datetime
import numpy as np

from utils.Utils import get_dataset_partitions_tf
from utils.Callbacks import lrfn, WeightDecayCallback

import tensorflow as tf
SEED = 57
tf.get_logger().setLevel('INFO')
tf.keras.utils.set_random_seed(SEED)

from tensorflow import keras
import keras_tuner as kt

import layers

config = None

DIM_NAMES = ['x', 'y']
TRANSFORMERV1 = True

# Hyperparamters
# Epsilon value for layer normalisation
LAYER_NORM_EPS = [1e-6]
# Dense layer units for landmarks
LANDMARK_UNITS = [512]
# final embedding and transformer embedding size
UNITS = [512, 1024]

# Transformer
NUM_BLOCKS = [2, 4, 6]
MLP_RATIO = 2
NUM_HEADS = 8
# Dropout
MLP_DROPOUT_RATIO = 0.30 # Transformer
CLASSIFIER_DROPOUT_RATIO = [0.10, 0.20]
# Initiailizers
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
# Activations
ACTIVATION = tf.keras.activations.gelu
# Learning Rate
LEARNING_RATE = [0.01, 0.001, 0.0001]
# Weight Decay
WEIGHT_DECAY = [0.0001, 0.00001]
# NUM_HEADS

# Indicies for slicing. 
FACE_IDXS = [0, 6, 7, 11, 12, 13, 14, 15, 17, 22, 23, 24, 25, 26, 30, 31, 
                     33, 37, 38, 39, 40, 41, 42, 56, 61, 62, 72, 73, 74, 76, 77, 
                     78, 80, 81, 82, 84, 86, 87, 88, 89, 90, 91, 95, 96, 110, 112, 
                     113, 122, 128, 130, 133, 144, 145, 146, 153, 154, 155, 157, 158, 
                     159, 160, 161, 163, 168, 173, 178, 179, 180, 181, 183, 184, 185, 
                     188, 189, 190, 191, 193, 196, 197, 232, 233, 243, 244, 245, 246, 
                     247, 249, 252, 253, 254, 255, 256, 259, 260, 263, 267, 268, 269, 
                     270, 271, 272, 286, 291, 292, 302, 303, 304, 306, 307, 308, 310, 
                     311, 312, 314, 316, 317, 318, 319, 320, 321, 324, 325, 339, 341, 
                     351, 357, 359, 362, 373, 374, 375, 380, 381, 382, 384, 385, 386, 
                     387, 388, 390, 398, 402, 403, 404, 405, 407, 408, 409, 412, 413, 
                     414, 415, 417, 419, 453, 463, 464, 465, 466, 467]
POSE_IDXS = np.arange(489, 514)
LEFT_HAND_IDXS = np.arange(468, 489)
RIGHT_HAND_IDXS = np.arange(522, 543)
# All landmarks that are used for modeling. 
LANDMARK_IDXS = np.concatenate((FACE_IDXS, POSE_IDXS, LEFT_HAND_IDXS, RIGHT_HAND_IDXS))
# Indicies after landmarks have been filtered. 
FACE_START = 0
LEFT_HAND_START = len(FACE_IDXS)
POSE_START = LEFT_HAND_START + len(LEFT_HAND_IDXS)
RIGHT_HAND_START = POSE_START + len(POSE_IDXS)
# Length of landmarks.
FACE_LEN = len(FACE_IDXS)
POSE_LEN = POSE_IDXS.size
LEFT_HAND_LEN = LEFT_HAND_IDXS.size
RIGHT_HAND_LEN = RIGHT_HAND_IDXS.size

def load_config_file(file_path):
    with open(file_path) as fp:
        config = json.load(fp)
        
    return config

def setup_wandb(config):
    # Setup Weights and Biases
    wandb.login()
    
    wandb.init(project='w251-asl-fp', 
               config=config,
               sync_tensorboard=True)
    
def load_data_from_fs():
    # Read in from local filesystem instead since reading from S3 takes too long. 
    X = np.load("./X.npy")
    y = np.load("./y.npy")
    NON_EMPTY_FRAME_IDXS = np.load("./NON_EMPTY_FRAME_IDXS.npy")
    
    return X, y, NON_EMPTY_FRAME_IDXS

def get_model(hp):
    global config
    # Inputs
    frames = tf.keras.layers.Input([config["INPUT_SIZE"], config["N_COLS"] * config["N_DIMS"]], dtype=tf.float32, name='FRAMES')
    non_empty_frame_idxs = tf.keras.layers.Input([config["INPUT_SIZE"]], dtype=tf.float32, name='NON_EMPTY_FRAME_IDXS')
    # Attention Mask
    mask = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)
    mask = tf.expand_dims(mask, axis=2)
    
    # Slice out face indicies       
    face = tf.slice(frames, [0, 0, FACE_START], [-1, config["INPUT_SIZE"], FACE_LEN * 2])    
     # Slice out left_hand indicies
    left_hand = tf.slice(frames, [0, 0, LEFT_HAND_START * 2], [-1, config["INPUT_SIZE"], LEFT_HAND_LEN * 2])
    # Slice out pose indicies
    pose = tf.slice(frames, [0, 0, POSE_START * 2], [-1, config["INPUT_SIZE"], POSE_LEN * 2])
    # Slice out right_hand indicies
    right_hand = tf.slice(frames, [0, 0, RIGHT_HAND_START * 2], [-1, config["INPUT_SIZE"], RIGHT_HAND_LEN * 2])

    # Embedding layer
    hp_landmark_units = hp.Choice('hp_landmark_units', values=LANDMARK_UNITS)
    hp_units = hp.Choice('hp_units', values=UNITS)
    embedding_layer = layers.Embedding(config["INPUT_SIZE"], hp_landmark_units, hp_landmark_units, hp_landmark_units, hp_units, ACTIVATION)
    x = embedding_layer(face, left_hand, right_hand, pose, non_empty_frame_idxs)
    
    # Encoder Transformer Blocks
    hp_layer_norm_eps = hp.Choice('layer_norm_eps', values=LAYER_NORM_EPS)
    hp_num_blocks = hp.Choice('num_blocks', values=NUM_BLOCKS)
    
    transformer_layer = layers.Transformer(hp_num_blocks, hp_layer_norm_eps, hp_units, MLP_RATIO, MLP_DROPOUT_RATIO, ACTIVATION)
    x = transformer_layer(x, mask)
    
    # Pooling
    x = tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
    # Classifier Dropout
    hp_classifier_dropout_ratio = hp.Choice('classifier_dropout_ratio', values=CLASSIFIER_DROPOUT_RATIO)
    
    x = tf.keras.layers.Dropout(hp_classifier_dropout_ratio)(x)
    # Classification Layer
    x = tf.keras.layers.Dense(config["NUM_CLASSES"], activation=tf.keras.activations.softmax, kernel_initializer=INIT_GLOROT_UNIFORM)(x)
    
    outputs = x
    
    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
    
    # Simple Categorical Crossentropy Loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Adam Optimizer with weight decay
    hp_learning_rate = hp.Choice('learning_rate', values=LEARNING_RATE)
    hp_weight_decay = hp.Choice('weight_decay', values=WEIGHT_DECAY)
    
    optimizer = tf.optimizers.AdamW(learning_rate=hp_learning_rate, weight_decay=hp_weight_decay)
    
    # TopK Metrics
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
    ]
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    return model


def main(args):
    if len(sys.argv) != 2:
        print("Pass path to config file.")
        
    global config
    config = load_config_file(args[1])
    
    setup_wandb(config)
    
    # Load dataset
    X, y, NON_EMPTY_FRAME_IDXS = load_data_from_fs()
    
    with tf.device('CPU'):
        dataset = tf.data.Dataset.from_tensor_slices(({"FRAMES": X, "NON_EMPTY_FRAME_IDXS": NON_EMPTY_FRAME_IDXS}, y))
        
    train, validation, test = get_dataset_partitions_tf(dataset, X.shape[0], train_split=0.8, val_split=0.1, 
                                                test_split=0.1, shuffle=True, shuffle_size=10000, seed=SEED)
    
    tuner = kt.Hyperband(get_model,
                     objective='val_acc',
                     max_epochs=config["N_EPOCHS"],
                     factor=3,
                     directory='tuning',
                     project_name='v2')
    
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=1)
    
    LR_SCHEDULE = [lrfn(step, num_warmup_steps=config["N_WARMUP_EPOCHS"], lr_max=config["LEARNING_RATE"], num_cycles=0.50) for step in range(config["N_EPOCHS"])]


    tuner.search(train.batch(config["TRAIN_BATCH_SIZE"]), validation_data=validation.batch(config["BATCH_SIZE_VAL"]), epochs=50, callbacks=[stop_early, tensorboard_callback, lr_callback, WeightDecayCallback(),])
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete.
    Best Hyperparameters:
    {best_hps}
    """)


if __name__ == "__main__":    
    main(sys.argv)
