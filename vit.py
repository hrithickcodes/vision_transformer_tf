import tensorflow as tf

from layers.class_token import ClassToken
from layers.multihead_attention import Multihead_attention
from layers.patch_embedding import PatchEmbeddings
from layers.positional_embedding import viTPositionalEmbedding
from layers.pwffn import PointWiseFeedForwardNetwork
from layers.transformer_encoder import TransformerEncoder


class viT(tf.keras.Model):
    def __init__(self,
                 image_size: tuple,
                 patch_size: int,
                 patch_embedding_dim: int,
                 mlp_units: int,
                 dropout_rate: float,
                 num_stacked_encoders: int,
                 num_attention_heads: int,
                 num_classes: int,
                 class_activation="linear",
                 layer_norm_epsilon=1e-6,
                 cls_token_init=tf.keras.initializers.HeNormal(),
                 positional_embedding_init=tf.keras.initializers.HeNormal(),
                 **kwargs):
        super(viT, self).__init__(**kwargs)
        assert len(image_size) == 3,\
            "image size should consist (image_height, image_width, image_channels)"
        self.patch_size = patch_size
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate
        self.num_stacked_encoders = num_stacked_encoders
        self.num_attention_heads = num_attention_heads
        self.num_classes = num_classes
        self.patch_embedding_dim = patch_embedding_dim
        self.image_height, self.image_width, self.image_channels = image_size

        # defining the layers
        self.patch_embedding = PatchEmbeddings(embedding_dimension=patch_embedding_dim,
                                               patch_size=self.patch_size,
                                               name="embedding")

        self.cls_layer = ClassToken(token_initializer=cls_token_init,
                                    name="class_token")

        self.pos_embedding = viTPositionalEmbedding(positional_embedding_init,
                                                    name="Transformer/posembed_input")
        # stacking up all the transformer encoder layers
        self.stacked_encoders = [TransformerEncoder(num_attention_heads=self.num_attention_heads,
                                                    mlp_units=self.mlp_units,
                                                    dropout_rate=self.dropout_rate,
                                                    name=f"Transformer/encoderblock_{layr}")
                                 for layr in range(self.num_stacked_encoders)]

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon,
                                                            name="Transformer/encoder_norm")

        self.get_CLS_token = tf.keras.layers.Lambda(lambda CLS: CLS[:, 0],
                                                    name="ExtractToken")
        self.dense_out = tf.keras.layers.Dense(self.num_classes,
                                               name="head",
                                               activation=class_activation)

    def call(self, input_tensor, training = False):
        # input_tensor: (batch_size, image_height, image_width, image_channels)
        x = self.patch_embedding(input_tensor)
        # reshaping
        x = tf.reshape(
            x, shape=(-1, x.shape[1] * x.shape[2], self.patch_embedding_dim))
        # input to CLS layer: (batch_size, patch_size * patch_size, patch_dimension)
        x = self.cls_layer(x)
        # adding positional embeddings
        x = self.pos_embedding(x)
        # input to posembedding layer: (batch_size, patch_size * patch_size + 1, patch_dimension)
        for tf_enc in self.stacked_encoders:
            # passing the input through all the transformer encoders
            x, _ = tf_enc(x, training=training)
        # input to layernorm layer: (batch_size, patch_size * patch_size + 1, patch_dimension)
        x = self.layernorm(x)
        # input to get_CLS_token layer: (batch_size, patch_size * patch_size + 1, patch_dimension)
        x = self.get_CLS_token(x)
        # input to out_dense layer: (batch_size, 1, patch_dimension)
        vit_logits = self.dense_out(x)
        # output shape: (batch_size, 1000)
        return vit_logits
