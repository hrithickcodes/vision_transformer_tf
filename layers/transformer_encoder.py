import tensorflow as tf
from layers.multihead_attention import Multihead_attention
from layers.pwffn import PointWiseFeedForwardNetwork

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, 
                 embedding_dimension,
                 num_attention_heads: int,
                 mlp_layer1_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embedding_dimension = embedding_dimension
        self.num_attention_heads = num_attention_heads
        self.mlp_layer1_units = mlp_layer1_units
        self.dropout_rate = dropout_rate
    
        self.mha = Multihead_attention(embedding_dimension = self.embedding_dimension,
                                        number_of_heads = self.num_attention_heads,
                                        name = "MultiHeadDotProductAttention_1")
        
        self.pwffn = PointWiseFeedForwardNetwork(layer1_units= self.mlp_layer1_units,
                                                 layer2_units = self.mha.embedding_dimension,
                                                 dropout_rate = self.dropout_rate)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        
    def call(self, input_tensor, training = False):
        x = self.layernorm1(input_tensor)
        x, attention_weights = self.mha(x)        
        x = self.dropout_layer(x, training=training)
        x = tf.math.add(x, input_tensor)
        y = self.layernorm2(x)
        y = self.pwffn(y)
        output_logits = tf.math.add(x, y)
        return output_logits, attention_weights
        
        
        
        