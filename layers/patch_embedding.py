import tensorflow as tf

class PatchEmbeddings(tf.keras.layers.Layer):
    def __init__(self, embedding_dimension: int, patch_size: int, **kwargs):
        super(PatchEmbeddings, self).__init__(**kwargs)
        self.embedding_dimension = embedding_dimension
        self.patch_size = patch_size

    def build(self, input_shape):
        # input_shape: (batch_size, image_height, image_width, image_channels)
        # getting the image_height and image_width
        self.image_height = input_shape[1]
        self.image_width = input_shape[2]
        assert (self.image_height % self.patch_size) + (self.image_height % self.patch_size) == 0, \
            "Image height and width should be divisible by the patch size"
        self.projection = tf.keras.layers.Conv2D(filters=self.embedding_dimension,
                                                 kernel_size=self.patch_size,
                                                 strides=self.patch_size,
                                                 padding='valid',
                                                 activation=None,
                                                 use_bias=True,
                                                 name="embedding",
                                                 kernel_initializer='glorot_uniform',
                                                 bias_initializer='glorot_uniform')

    def call(self, input_tensor):
        output_embedding_logits = self.projection(input_tensor)
        return output_embedding_logits
