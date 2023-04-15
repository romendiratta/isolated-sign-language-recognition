import tensorflow as tf

class TransformerV2(tf.keras.layers.Layer):
    def __init__(self, t_input_shape, num_heads, units, dropout_rate, layer_norm_eps):
        super(TransformerV2, self).__init__()
        self.t_input_shape = t_input_shape
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.units = units
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            't_input_shape': self.t_input_shape,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'layer_norm_eps': self.layer_norm_eps,
            'units': self.units,
        })
        return config

    def call(self, inputs):
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.t_input_shape[-1])(inputs, inputs)
        attn_output = tf.keras.layers.Dropout(self.dropout_rate)(attn_output)
        out1 = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_eps)(inputs + attn_output)
        ff_output = tf.keras.layers.Dense(self.units, activation='relu')(out1)
        ff_output = tf.keras.layers.Dropout(self.dropout_rate)(ff_output)
        ff_output = tf.keras.layers.Dense(self.t_input_shape[-1])(ff_output)  # Add this line to match the dimensions
        out2 = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_eps)(out1 + ff_output)
        return out2
    
    
#     def build(self, input_shape):
#         self.model = x
        
#         for _ in range(num_encoders):
#             x = transformer_encoder(input_shape, num_heads, ff_dim, dropout_rate)(x)
#             encoder_input_shape = x.shape[1:]  # Update the input shape for the next encoder

#     def call(self, inputs):
#         x = inputs
#         for _ in range(self.num_blocks):
#             x = transformer_encoder(inputs.shape, self.num_heads, self.units, self.dropout_rate)(x)
#             encoder_input_shape = x.shape[1:]  # Update the input shape for the next encoder
