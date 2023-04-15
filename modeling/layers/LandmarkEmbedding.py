import tensorflow as tf

class LandmarkEmbedding(tf.keras.layers.Layer):
    def __init__(self, units, name, activation):
        super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
        self.UNITS = units
        self.ACTIVATION = activation
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'UNITS': self.UNITS,
            'ACTIVATION': self.ACTIVATION,
        })
        return config
        
    def build(self, input_shape):
        # Embedding for missing landmark in frame, initizlied with zeros
        self.empty_embedding = self.add_weight(
            name=f'{self.name}_empty_embedding',
            shape=[self.UNITS],
            initializer=tf.keras.initializers.constant(0.0),
        )
        # Embedding
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(self.UNITS, name=f'{self.name}_dense_1', use_bias=False, 
                                  kernel_initializer=tf.keras.initializers.glorot_uniform, activation=self.ACTIVATION),
            tf.keras.layers.Dense(self.UNITS, name=f'{self.name}_dense_2', use_bias=False, kernel_initializer=tf.keras.initializers.he_uniform),
        ], name=f'{self.name}_dense')

    def call(self, x):
        return tf.where(
                # Checks whether landmark is missing in frame
                tf.reduce_sum(x, axis=2, keepdims=True) == 0,
                # If so, the empty embedding is used
                self.empty_embedding,
                # Otherwise the landmark data is embedded
                self.dense(x),
            )