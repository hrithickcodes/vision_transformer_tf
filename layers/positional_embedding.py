import tensorflow as tf

class viTPositionalEmbedding(tf.keras.layers.Layer):
    """Adds a Learnable Positional Encoding layer.
    """
    def __init__(self, pe_initializer, **kwargs):
        super(viTPositionalEmbedding, self).__init__(**kwargs)
        self.pe_initializer = pe_initializer

    def build(self, input_shape):
        # input_shape: (batch_size, number_of_tokens + cls_token, embedding_dimension)
        # getting the number of total tokens
        self.num_tokens = input_shape[1]
        # getting the embedding dimensions
        self.embedding_dimension = input_shape[-1]
        # shape of the positional embeddings, shape : (1, number_of_tokens + cls_token, embedding_dimension)
        self.learnable_pe_shape = (1, self.num_tokens, self.embedding_dimension)
        self.learnable_pe = tf.Variable(name="pos_embedding",
                                        initial_value=self.pe_initializer(self.learnable_pe_shape),
                                        dtype="float32",
                                        trainable=True)
        
    def call(self, input_tensor):
        # shape of image_tensor : (batch_size, number_of_tokens + cls_token, embedding_dimension)
        # casting the positional embeddings to the same dtype of the inputs
        self.pe_var = tf.cast(self.learnable_pe, dtype=input_tensor.dtype)
        # adding the pe to the input tensor
        output_logits = tf.math.add(input_tensor, self.pe_var)
        return output_logits