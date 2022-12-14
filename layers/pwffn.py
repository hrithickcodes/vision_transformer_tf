import tensorflow as tf 

class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):
    """The Multilayer Perceptron layer used in the transformer encoder
    """
    def __init__(self, units, dropout_rate, **kwargs):
        super(PointWiseFeedForwardNetwork, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.dense_1 = tf.keras.layers.Dense(units = self.units,
                                            activation="linear",
                                            name=f"{self.name}/Dense_0")        
        self.dropout_1 = tf.keras.layers.Dropout(self.dropout_rate)
        
        self.dense_2 = tf.keras.layers.Dense(units = input_shape[-1],
                                             name=f"{self.name}/Dense_1")
        self.dropout_2 = tf.keras.layers.Dropout(self.dropout_rate)
        
    def call(self, input_tensor):
        x = self.dense_1(input_tensor)
        x = tf.keras.activations.gelu(x, approximate=False)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        return x
    