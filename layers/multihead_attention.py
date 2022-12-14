import tensorflow as tf

class Multihead_attention(tf.keras.layers.Layer):
    def __init__(self, number_of_heads: int, **kwargs):
        super(Multihead_attention, self).__init__(**kwargs)
        self.number_of_heads = number_of_heads
    
    def scaled_dot_product_attention(self, query, key, value):
        # Matrix multiplication of the Query and the keys
        # shape: [batch size, number of attention heads, sequence length, sequnce length]
        Q_K_matmul = tf.matmul(a = query,
                                b = key,
                                transpose_b= True )

        # getting the embedding dimension from the keys
        d = tf.shape(key)[-1]
        d = tf.cast(d, tf.float32)
    
        # scaling the matrix with the embedding dimension for stable gradients
        raw_scores = tf.divide(Q_K_matmul, tf.math.sqrt(d))
        # comuting the softmax with respect to the last dimension
        # last sequence length axis 
        # shape: [batch size, numbe of attention heads, sequence length, sequence length (softmax axis)]
        attention_weights = tf.nn.softmax(raw_scores, axis = -1)

        # weighing the attention score with the values
        # shape: [batch size, numbe of attention heads, sequence length, head_dimension]
        output_logits = tf.matmul(attention_weights, value)

        return output_logits, attention_weights

        
    def build(self, input_shape):
        self.embedding_dimension = input_shape[-1]
        self.head_dimension = self.embedding_dimension // self.number_of_heads
        assert (self.head_dimension * self.number_of_heads) == self.embedding_dimension , \
            "Embedding dimension should be divisible by the numbe of heads"

        self.wQ = tf.keras.layers.Dense(self.embedding_dimension, name = "query")
        self.wK = tf.keras.layers.Dense(self.embedding_dimension, name = "key")
        self.wV = tf.keras.layers.Dense(self.embedding_dimension, name = "value")
        self.FFN = tf.keras.layers.Dense(self.embedding_dimension, name = "out")

    def split_to_heads(self, input_tensor, batch_size):
        splitted_shape = (batch_size, -1, self.number_of_heads, self.head_dimension)
        splitted_input_tensor = tf.reshape(input_tensor, splitted_shape)
        return tf.transpose(splitted_input_tensor, perm=[0, 2, 1, 3])


    def call(self, input_tensor):
        self.batch_size = tf.shape(input_tensor)[0]
        self.batch_size = tf.cast(self.batch_size, tf.int64)
        
        # learning the query, key and value matrices
        queries = self.wQ(input_tensor) 
        keys = self.wK(input_tensor)  
        values = self.wV(input_tensor)


        # splitting the last embedding dimension with number_of_heads, head_dimension
        # shape of all the tensors: [batch_size, num_attention_heads, sequence_length, head_dimension]
        self.queries = self.split_to_heads(queries, self.batch_size)
        self.keys = self.split_to_heads(keys, self.batch_size)
        self.values = self.split_to_heads(values, self.batch_size)

        # Computing logits and attention weights using scaled dot product attention
        logits, attention_weights = self.scaled_dot_product_attention(self.queries,
                                                                        self.keys,
                                                                        self.values)

        # Earlier [batch size, numbe of attention heads, sequence length, head_dimension]
        # after transpose: [batch_size, sequence length, num_attention heads, head_dimenson]
        logits = tf.transpose(logits, perm=[0, 2, 1, 3])  
        # number_of_heads, head_dimension -> embedding_dimension, concating the logits
        # concatenating to [batch_size, sequence length, num_attention heads * head_dimenson]
        concated_logits = tf.reshape(logits,
                                    (self.batch_size, -1, self.embedding_dimension)) 
    
        output_logits = self.FFN(concated_logits)  

        return output_logits, attention_weights
    
    

