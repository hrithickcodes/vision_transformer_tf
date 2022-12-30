import tensorflow as tf


class ClassToken(tf.keras.layers.Layer):
    """The Class Token class concatenates the cls_token with the input_tensor
    """

    def __init__(self,
                 embedding_dimension,
                 **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.embedding_dimension = embedding_dimension
        self.cls_token_tensor = tf.Variable(name="cls",
                                            initial_value = tf.random_normal_initializer(stddev=0.06)
                                            (shape = (1, 1, self.embedding_dimension)),
                                            trainable=True)

    def call(self, input_tensor):
        self.batch_size = tf.shape(input_tensor)[0]
        self.cls_token_broadcasted = tf.broadcast_to(self.cls_token_tensor,
                                                     shape=[self.batch_size, 1, self.embedding_dimension])
        self.cls_token = tf.cast(
            self.cls_token_broadcasted, dtype=input_tensor.dtype)
        self.output_tensor = tf.concat([self.cls_token, input_tensor], axis=1)
        return self.output_tensor
