import tensorflow as tf
from layers.LandmarkEmbedding import LandmarkEmbedding

class Embedding(tf.keras.Model):
    def __init__(self, input_size, face_units, hands_units, pose_units, units, activation):
        super(Embedding, self).__init__()
        self.INPUT_SIZE = input_size
        self.FACE_UNITS = face_units
        self.HANDS_UNITS = hands_units
        self.POSE_UNITS = pose_units
        self.UNITS = units
        self.ACTIVATION = activation
        
    # def get_diffs(self, l):
    #     S = l.shape[2]
    #     other = tf.expand_dims(l, 3)
    #     other = tf.repeat(other, S, axis=3)
    #     other = tf.transpose(other, [0,1,3,2])
    #     diffs = tf.expand_dims(l, 3) - other
    #     diffs = tf.reshape(diffs, [-1, config["INPUT_SIZE"], S*S])
    #     return diffs

    def build(self, input_shape):
        # Positional Embedding, initialized with zeros
        self.positional_embedding = tf.keras.layers.Embedding(self.INPUT_SIZE+1, self.UNITS, embeddings_initializer=tf.keras.initializers.constant(0.0))
        # Embedding layer for Landmarks
        self.face_embedding = LandmarkEmbedding(self.FACE_UNITS, 'lips', self.ACTIVATION)
        self.left_hand_embedding = LandmarkEmbedding(self.HANDS_UNITS, 'left_hand', self.ACTIVATION)
        self.right_hand_embedding = LandmarkEmbedding(self.HANDS_UNITS, 'right_hand', self.ACTIVATION)
        self.pose_embedding = LandmarkEmbedding(self.POSE_UNITS, 'pose', self.ACTIVATION)
        # Landmark Weights
        self.landmark_weights = tf.Variable(tf.zeros([4], dtype=tf.float32), name='landmark_weights')
        # Fully Connected Layers for combined landmarks
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(self.UNITS, name='fully_connected_1', use_bias=False, 
                                  kernel_initializer=tf.keras.initializers.glorot_uniform, activation=self.ACTIVATION),
            tf.keras.layers.Dense(self.UNITS, name='fully_connected_2', use_bias=False, kernel_initializer=tf.keras.initializers.he_uniform),
        ], name='fc')


    def call(self, face0, left_hand0, right_hand0, pose0, non_empty_frame_idxs, training=False):
        # Face
        face_embedding = self.face_embedding(face0)
        # Left Hand
        left_hand_embedding = self.left_hand_embedding(left_hand0)
        # Right Hand
        right_hand_embedding = self.right_hand_embedding(right_hand0)
        # Pose
        pose_embedding = self.pose_embedding(pose0)
        # Merge Embeddings of all landmarks with mean pooling
        x = tf.stack((face_embedding, left_hand_embedding, right_hand_embedding, pose_embedding), axis=3)
        # Merge Landmarks with trainable attention weights
        x = x * tf.nn.softmax(self.landmark_weights)
        x = tf.reduce_sum(x, axis=3)
        # Fully Connected Layers
        x = self.fc(x)
        # Add Positional Embedding
        normalised_non_empty_frame_idxs = tf.where(
            tf.math.equal(non_empty_frame_idxs, -1.0),
            self.INPUT_SIZE,
            tf.cast(
                non_empty_frame_idxs / tf.reduce_max(non_empty_frame_idxs, axis=1, keepdims=True) * self.INPUT_SIZE,
                tf.int32,
            ),
        )
        x = x + self.positional_embedding(normalised_non_empty_frame_idxs)
        
        return x