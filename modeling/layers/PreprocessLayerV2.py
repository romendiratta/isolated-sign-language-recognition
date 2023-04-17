import tensorflow as tf
# import numpy as np

class PreprocessLayerV2(tf.keras.layers.Layer):
    def __init__(self, INPUT_SIZE):
        super(PreprocessLayerV2, self).__init__()
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
        
        self.HAND_IDXS = tf.constant(tf.concat([self.LEFT_HAND_IDXS, self.RIGHT_HAND_IDXS], 0), dtype=tf.int32)
            
        # All landmarks that are used for modeling. 
        self.LANDMARK_IDXS = tf.constant(tf.concat([self.FACE_IDXS, self.POSE_IDXS, self.LEFT_HAND_IDXS, self.RIGHT_HAND_IDXS], 0), dtype=tf.int32)
        
        # Indicies after landmarks have been filtered. 
        self.FACE_START = tf.constant(0, dtype=tf.int32)
        self.LEFT_HAND_START = tf.constant(len(self.FACE_IDXS), dtype=tf.int32)
        self.POSE_START = tf.constant(self.LEFT_HAND_START + len(self.LEFT_HAND_IDXS), dtype=tf.int32)
        self.RIGHT_HAND_START = tf.constant(self.POSE_START + len(self.POSE_IDXS), dtype=tf.int32)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'INPUT_SIZE': self.INPUT_SIZE,
        })
        return config
    
    def get_mean_std(self, LIPS_IDXS, data):
                
        xs = tf.reshape(tf.transpose(data, [2,0,1]), [2,tf.size(LIPS_IDXS)*tf.shape(data)[0]])[0]
        ys = tf.reshape(tf.transpose(data, [2,0,1]), [2,tf.size(LIPS_IDXS)*tf.shape(data)[0]])[1]
            
        LIPS_MEAN_X = tf.math.reduce_mean(xs)
        LIPS_STD_X = tf.math.reduce_std(xs)
        LIPS_MEAN_Y = tf.math.reduce_mean(ys)
        LIPS_STD_Y = tf.math.reduce_std(ys)

        LIPS_MEAN = tf.stack([LIPS_MEAN_X, LIPS_MEAN_Y])
        LIPS_STD = tf.stack([LIPS_STD_X, LIPS_STD_Y])

        return LIPS_MEAN, LIPS_STD
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 543, 2], dtype=tf.float32),),)
    def call(self, data):
        
        # Filter Out Frames With Empty Hand Data
        frames_hands_nansum = tf.experimental.numpy.nanmean(tf.gather(data, self.HAND_IDXS, axis=1), axis=[1,2])
        non_empty_frames_idxs = tf.where(frames_hands_nansum > 0)
        non_empty_frames_idxs = tf.squeeze(non_empty_frames_idxs, axis=1)
        data = tf.gather(data, non_empty_frames_idxs, axis=0)
        
        # Cast Indices in float32 to be compatible with Tensorflow Lite
        non_empty_frames_idxs = tf.cast(non_empty_frames_idxs, tf.float32) 
        
        N_FRAMES = tf.shape(data)[0]
        data = tf.gather(data, self.LANDMARK_IDXS, axis=1)
        
        # Slice out face indicies, normalize across batch.        
        face = tf.slice(data, [0, self.FACE_START, 0], [N_FRAMES, self.LEFT_HAND_START, 2])
        face_mean, face_std = self.get_mean_std(self.FACE_IDXS, face)
        face = tf.where(
                    tf.math.equal(face, 0.0),
                    0.0,
                    (face - face_mean) / face_std,
                )
        # face = tf.keras.utils.normalize(face, axis=-1, order=2)
        
#         # Slice out left_hand indicies, normalize across batch.
        left_hand = tf.slice(data, [0, self.LEFT_HAND_START, 0], [N_FRAMES, self.POSE_START-self.LEFT_HAND_START, 2])
        # left_hand = tf.keras.utils.normalize(left_hand, axis=1, order=2)
        # left_hand = tf.linalg.normalize(left_hand, axis=1)
        left_hand_mean, left_hand_std = self.get_mean_std(self.LEFT_HAND_IDXS, left_hand)
        left_hand = tf.where(
                    tf.math.equal(left_hand, 0.0),
                    0.0,
                    (left_hand - left_hand_mean) / left_hand_std,
                )
        
#         # Slice out pose indicies, normalize across batch.
        pose = tf.slice(data, [0, self.POSE_START, 0], [N_FRAMES, self.RIGHT_HAND_START-self.POSE_START, 2])
        # pose = tf.keras.utils.normalize(pose, axis=1, order=2)
        pose_mean, pose_std = self.get_mean_std(self.POSE_IDXS, pose)
        pose = tf.where(
                    tf.math.equal(pose, 0.0),
                    0.0,
                    (pose - pose_mean) / pose_std,
                )
        
#         # Slice out right_hand indicies, normalize across batch.
        right_hand = tf.slice(data, [0, self.RIGHT_HAND_START, 0], [N_FRAMES, tf.shape(data)[1] - self.RIGHT_HAND_START, 2])
#         # right_hand = tf.keras.utils.normalize(right_hand, axis=1, order=2)
        right_hand_mean, right_hand_std = self.get_mean_std(self.RIGHT_HAND_IDXS, right_hand)
        right_hand = tf.where(
                    tf.math.equal(right_hand, 0.0),
                    0.0,
                    (right_hand - right_hand_mean) / right_hand_std,
                )
        
        
        # Concat landmarks back into same frame.
        data = tf.concat([face, left_hand, pose, right_hand], 1)
        
        
        # Video fits in self.INPUT_SIZE
        if N_FRAMES < self.INPUT_SIZE: # Number of frames we want
            # Attention mask for frames that contain data. 
            
            non_empty_frames_idxs = tf.pad(non_empty_frames_idxs, [[0, self.INPUT_SIZE-N_FRAMES]], constant_values=-1)
            # non_empty_frames_idxs = tf.pad(tf.range(0, N_FRAMES, 1), [[0, self.INPUT_SIZE-N_FRAMES]], constant_values=-1)
            data = tf.pad(data, [[0, self.INPUT_SIZE-N_FRAMES], [0,0], [0,0]], constant_values=0)
            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            # Reshape into (Number of desired frames, (Number of landmarks * 2))
            data = tf.reshape(data, [self.INPUT_SIZE, tf.shape(data)[1] * 2])
            return data, non_empty_frames_idxs
        # Video needs to be downsampled to INPUT_SIZE
        else:
            # Downsample video using nearest interpolation method. 
            data = tf.image.resize(data, size=(self.INPUT_SIZE, data.shape[1]), method='nearest')
            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            # Reshape into (Number of desired frames, (Number of landmarks * 2)).
            data = tf.reshape(data, [self.INPUT_SIZE, tf.shape(data)[1] * 2])
            # Create attention mask with all frames. 
            non_empty_frames_idxs = tf.range(0, self.INPUT_SIZE, 1, dtype=tf.float32)
            return data, non_empty_frames_idxs