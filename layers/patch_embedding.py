import tensorflow as tf

class PatchEmbeddings(tf.keras.layers.Layer):
    def __init__(self, embedding_dimension: int, patch_size: int, **kwargs):
        super(PatchEmbeddings, self).__init__(**kwargs)
        self.embedding_dimension = embedding_dimension
        self.patch_size = patch_size
        
        self.projection = tf.keras.layers.Conv2D(filters=self.embedding_dimension,
                                                 kernel_size=self.patch_size,
                                                 strides=self.patch_size,
                                                 padding='valid',
                                                 name="embedding")

    def call(self, input_tensor):
        output_embedding_logits = self.projection(input_tensor)
        return output_embedding_logits
