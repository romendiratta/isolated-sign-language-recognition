import tensorflow as tf

class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self, INPUT_SIZE):
        super(PreprocessLayer, self).__init__()
        self.INPUT_SIZE = INPUT_SIZE
        # Indicies in original data. 
        self.FACE_IDXS = tf.constant([0, 6, 7, 11, 12, 13, 14, 15, 17, 22, 23, 24, 25, 26, 30, 31, 
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
                     414, 415, 417, 419, 453, 463, 464, 465, 466, 467], dtype=tf.int32)
        self.POSE_IDXS = tf.constant(tf.range(489, 514, delta=1, dtype=tf.int32))
        self.LEFT_HAND_IDXS = tf.constant(tf.range(468, 489, delta=1, dtype=tf.int32))
        self.RIGHT_HAND_IDXS = tf.constant(tf.range(522, 543, delta=1, dtype=tf.int32))
            
        # All landmarks that are used for modeling. 
        self.LANDMARK_IDXS = tf.constant(tf.concat([self.FACE_IDXS, self.POSE_IDXS, self.LEFT_HAND_IDXS, self.RIGHT_HAND_IDXS], 0), dtype=tf.int32)
        
        # Indicies after landmarks have been filtered. 
        self.FACE_START = tf.constant(0, dtype=tf.int32)
        self.LEFT_HAND_START = tf.constant(len(self.FACE_IDXS), dtype=tf.int32)
        self.POSE_START = tf.constant(self.LEFT_HAND_START + len(self.LEFT_HAND_IDXS), dtype=tf.int32)
        self.RIGHT_HAND_START = tf.constant(self.POSE_START + len(self.POSE_IDXS), dtype=tf.int32)
    
    # @tf.function(
    #     input_signature=(tf.TensorSpec(shape=[None, 543, 2], dtype=tf.float32), ),
    # )
    def call(self, data):
        N_FRAMES = tf.shape(data)[0]
        data = tf.gather(data, self.LANDMARK_IDXS, axis=2)
        
        # Slice out face indicies, normalize across batch.        
        face = tf.slice(data, [0, self.FACE_START, 0], [N_FRAMES, self.LEFT_HAND_START, 2])
        face = tf.keras.utils.normalize(face, axis=1, order=2)
        
        # Slice out left_hand indicies, normalize across batch.
        left_hand = tf.slice(data, [0, self.LEFT_HAND_START, 0], [N_FRAMES, self.POSE_START-self.LEFT_HAND_START, 2])
        left_hand = tf.keras.utils.normalize(left_hand, axis=1, order=2)
        
        # Slice out pose indicies, normalize across batch.
        pose = tf.slice(data, [0, self.POSE_START, 0], [N_FRAMES, self.RIGHT_HAND_START-self.POSE_START, 2])
        pose = tf.keras.utils.normalize(pose, axis=1, order=2)
        
        # Slice out right_hand indicies, normalize across batch.
        right_hand = tf.slice(data, [0, self.RIGHT_HAND_START, 0], [N_FRAMES, tf.shape(data)[2] - self.RIGHT_HAND_START, 2])
        right_hand = tf.keras.utils.normalize(right_hand, axis=1, order=2)
        
        # Concat landmarks back into same frame.
        data = tf.concat([face, left_hand, pose, right_hand], 1)
        
        # Video fits in self.INPUT_SIZE
        if N_FRAMES < self.INPUT_SIZE: # Number of frames we want
            # Attention mask for frames that contain data. 
            non_empty_frames_idxs = tf.pad(tf.range(0, N_FRAMES, 1), [[0, self.INPUT_SIZE-N_FRAMES]], constant_values=-1)
            data = tf.pad(data, [[0, self.INPUT_SIZE-N_FRAMES], [0,0], [0,0]], constant_values=-1)
            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0, data)
            # Reshape into (Number of desired frames, (Number of landmarks * 2))
            data = tf.reshape(data, [self.INPUT_SIZE, tf.shape(data)[1] * 2])
            return data, non_empty_frames_idxs
        # Video needs to be downsampled to INPUT_SIZE
        else:
            # Downsample video using nearest interpolation method. 
            data = tf.image.resize(data, size=(self.INPUT_SIZE, data.shape[1]), method='nearest')
            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0, data)
            # Reshape into (Number of desired frames, (Number of landmarks * 2)).
            data = tf.reshape(data, [self.INPUT_SIZE, tf.shape(data)[1] * 2])
            # Create attention mask with all frames. 
            non_empty_frames_idxs = tf.range(0, self.INPUT_SIZE, 1)
            return data, non_empty_frames_idxs