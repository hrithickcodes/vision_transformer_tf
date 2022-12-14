import tensorflow as tf 

class ClassToken(tf.keras.layers.Layer):
    """The Class Token class concatenates the cls_token with the input_tensor
    """
    
    def __init__(self, token_initializer, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.token_initializer = token_initializer

    def build(self, input_shape):
        #  input_shape: (batch_size, number_of_tokens, embedding_dimension)
        # getting the embedding dimension
        self.embedding_dimension = input_shape[-1]
        # initial shape of the class_token tensor, later on we will broadcast to the batch size along the first dimension
        self.cls_token_init_shape = (1, 1, self.embedding_dimension)
        # instead of any random initialization we can use techniques like he_normal, glorot_uniform etc
        self.cls_token_init_value = self.token_initializer(shape=self.cls_token_init_shape)
        # initializing the class token tensor
        self.cls_token_tensor = tf.Variable(name="cls",
                                            initial_value=self.cls_token_init_value,
                                            trainable=True)
    
    def call(self, input_tensor):
        # input_tesnor shape: (batch_size, number_of_tokens, embedding_dimension)
        self.batch_size = tf.shape(input_tensor)[0]
        # brodcasting the class token with the batch_size along the first dimension
        self.cls_token_broadcasted = tf.broadcast_to(self.cls_token_tensor,
                                                    shape = [self.batch_size, 1, self.embedding_dimension])
        # dtype should be same as we need to concat later on
        self.cls_token = tf.cast(self.cls_token_broadcasted, dtype=input_tensor.dtype)
        # concatenating the class_token with the input_tensor
        self.output_tensor = tf.concat([self.cls_token, input_tensor], axis = 1)
        # output_shape : (batch_size, number_of_tokens + 1, embedding_dimension)
        return self.output_tensor 