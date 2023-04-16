import tensorflow as tf

class TransformerV2(tf.keras.layers.Layer):
    def __init__(self, t_input_shape, num_heads, units, dropout_rate, layer_norm_eps, num_blocks):
        super(TransformerV2, self).__init__()
        self.t_input_shape = t_input_shape
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.units = units
        self.num_blocks = num_blocks
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            't_input_shape': self.t_input_shape,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'layer_norm_eps': self.layer_norm_eps,
            'units': self.units,
            'num_blocks': self.num_blocks
        })
        return config

    
    def build(self, input_shape):
        self.mhsa = []
        self.mha_dropouts = []
        self.ln_1s = []
        self.dense_1s = []
        self.mlp_droupouts = []
        self.dense_2s = []
        self.ln_2s = []
        for i in range(self.num_blocks):
            self.mhsa.append(tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.t_input_shape[-1]))
            self.mha_dropouts.append(tf.keras.layers.Dropout(self.dropout_rate))
            self.ln_1s.append(tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_eps))
            self.dense_1s.append(tf.keras.layers.Dense(self.units, activation='relu'))
            self.mlp_droupouts.append(tf.keras.layers.Dropout(self.dropout_rate))
            self.dense_2s.append(tf.keras.layers.Dense(self.t_input_shape[-1]))
            self.ln_1s.append(tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_eps))
            
    def call(self, x):
        for mha, dropout1, ln_1, dense1, dropout2, dense2, ln_2 in zip(self.mhsa, self.mha_dropouts, 
                                                                       self.ln_1s, self.dense_1s, 
                                                                       self.mlp_droupouts, self.dense_2s, self.ln_2s):
            attn_output = mha(x, x)
            attn_output = dropout1(attn_output)
            out1 = ln_1(inputs + attn_output)
            ff_output = dense1(out1)
            ff_output = dropout2(ff_output)
            ff_output = dense2(ff_output)
            x = ln_2(out1 + ff_output)

        return x