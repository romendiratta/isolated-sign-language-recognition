import tensorflow as tf
from layers.MultiHeadAttention import MultiHeadAttention


# Full Transformer
class Transformer(tf.keras.layers.Layer):
    def __init__(self, num_blocks, layer_norm_eps, units, mlp_ratio, mlp_dropout_ratio, activation):
        super(Transformer, self).__init__(name='transformer')
        self.NUM_BLOCKS = num_blocks
        self.LAYER_NORM_EPS = layer_norm_eps
        self.UNITS = units
        self.MLP_RATIO = mlp_ratio
        self.MLP_DROPOUT_RATIO = mlp_dropout_ratio
        self.ACTIVATION = activation
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'NUM_BLOCKS': self.NUM_BLOCKS,
            'LAYER_NORM_EPS': self.LAYER_NORM_EPS,
            'MLP_RATIO': self.MLP_RATIO,
            'MLP_DROPOUT_RATIO': self.MLP_DROPOUT_RATIO,
            'UNITS': self.UNITS,
            'ACTIVATION': self.ACTIVATION,
        })
        return config
    
    def build(self, input_shape):
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        # Make Transformer Blocks
        for i in range(self.NUM_BLOCKS):
            # First Layer Normalisation
            self.ln_1s.append(tf.keras.layers.LayerNormalization(epsilon=self.LAYER_NORM_EPS))
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(self.UNITS, 4))
            # Second Layer Normalisation
            self.ln_2s.append(tf.keras.layers.LayerNormalization(epsilon=self.LAYER_NORM_EPS))
            # Multi Layer Perception
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(self.UNITS * self.MLP_RATIO, activation=self.ACTIVATION, 
                                      kernel_initializer=tf.keras.initializers.glorot_uniform),
                tf.keras.layers.Dropout(self.MLP_DROPOUT_RATIO),
                tf.keras.layers.Dense(self.UNITS, kernel_initializer=tf.keras.initializers.he_uniform),
            ]))
        
    def call(self, x, attention_mask):
        # Iterate input over transformer blocks
        for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
            x1 = ln_1(x)
            attention_output = mha(x1, attention_mask)
            x2 = x1 + attention_output
            x3 = ln_2(x2)
            x3 = mlp(x3)
            x = x3 + x2
    
        return x