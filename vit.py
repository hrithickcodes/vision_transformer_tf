import tensorflow as tf

from layers.class_token import ClassToken
from layers.multihead_attention import Multihead_attention
from layers.patch_embedding import PatchEmbeddings
from layers.positional_embedding import viTPositionalEmbedding
from layers.pwffn import PointWiseFeedForwardNetwork
from layers.transformer_encoder import TransformerEncoder
from utils.general import load_config

class viT(tf.keras.Model):
    def __init__(self,
                 vit_size,
                 num_classes = 1000,
                 class_activation="softmax",
                 config_path="vit_architectures.yaml",
                 **kwargs):
        super(viT, self).__init__(**kwargs)
        self.vit_size = vit_size
        self.num_classes = num_classes         

        self.vit_attr = load_config(config_path)[self.vit_size]

        self.patch_size = self.vit_attr["patch_size"]
        self.mlp_layer1_units = self.vit_attr["units_in_mlp"]
        self.dropout_rate = self.vit_attr["dropout_rate"]
        self.num_stacked_encoders = self.vit_attr["encoder_layers"]
        self.num_attention_heads = self.vit_attr["attention_heads"]
        self.patch_embedding_dim = self.vit_attr["patch_embedding_dim"]
        self.image_size = self.vit_attr["image_size"]
        self.image_height, self.image_width, self.image_channels = self.image_size
        

        assert len(self.image_size) == 3,\
            "image size should consist (image_height, image_width, image_channels)"

        self.class_activation = class_activation
        self.num_classes = self.num_classes

        self.patch_embedding = PatchEmbeddings(embedding_dimension=self.patch_embedding_dim,
                                               patch_size=self.patch_size,
                                               name="embedding")

        self.cls_layer = ClassToken(name="class_token",
                                    embedding_dimension=self.patch_embedding_dim)

        self.pos_embedding = viTPositionalEmbedding(
            num_of_tokens=(self.image_height // self.patch_size) *
            (self.image_width // self.patch_size) + 1,
            embedding_dimension=self.patch_embedding_dim,
            name="Transformer/posembed_input")

        self.stacked_encoders = [TransformerEncoder(embedding_dimension=self.patch_embedding_dim,
                                                    num_attention_heads=self.num_attention_heads,
                                                    mlp_layer1_units=self.mlp_layer1_units,
                                                    dropout_rate=self.dropout_rate,
                                                    name=f"Transformer/encoderblock_{layr}")
                                 for layr in range(self.num_stacked_encoders)]

        self.layernorm = tf.keras.layers.LayerNormalization(
            name="Transformer/encoder_norm")

        self.get_CLS_token = tf.keras.layers.Lambda(lambda CLS: CLS[:, 0],
                                                    name="ExtractToken")
        self.dense_out = tf.keras.layers.Dense(self.num_classes,
                                               name="head",
                                               activation=self.class_activation)

        self.build(
            [1, self.image_height, self.image_width, self.image_channels])

    def call(self, input_tensor, training=False):
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
        x = self.dense_out(x)
        # output shape: (batch_size, 1000)
        return x

    def preprocess_input(self, image):
        image = tf.image.resize(image, size=(
            self.image_height, self.image_width))
        output_tensor = tf.keras.applications.imagenet_utils.preprocess_input(image,
                                                                              data_format=None,
                                                                              mode="tf")
        return output_tensor

    def from_pretrained(self,
                        pretrained_top=True):
        from utils.npz_weights_loader import (get_pretrained_encoders_num,
                                              load_npz_file,
                                              reshape_mha_matrix,
                                              reshape_mha_out_matrix)
        params = load_npz_file(pretrained_size=self.vit_size)
        num_transformer_encoder = get_pretrained_encoders_num(
            pretrained_param_dict=params)

        source_used_keys = []

        self.cls_layer.set_weights([params["cls"]])

        source_used_keys.extend(['cls'])

        self.pos_embedding.set_weights(
            [params["Transformer/posembed_input/pos_embedding"]])

        source_used_keys.extend(['Transformer/posembed_input/pos_embedding'])

        patch_emb_source_keys = [
            f"embedding/{name}" for name in ["kernel", "bias"]]

        self.patch_embedding.set_weights(
            [params[key] for key in patch_emb_source_keys])

        source_used_keys.extend(patch_emb_source_keys)

        all_source_keys = list(params.keys())
        for i_encoder, target_encoder in enumerate(self.stacked_encoders):
            layer_name = f"Transformer/encoderblock_{i_encoder}"

            layernorm1_source_key = [
                f"{layer_name}/LayerNorm_0/{name}" for name in ["scale", "bias"]]
            layernorm2_source_key = [
                f"{layer_name}/LayerNorm_2/{name}" for name in ["scale", "bias"]]
            target_encoder.layernorm1.set_weights(
                [params[key] for key in layernorm1_source_key])
            target_encoder.layernorm2.set_weights(
                [params[key] for key in layernorm2_source_key])

            source_used_keys.extend(
                layernorm1_source_key + layernorm2_source_key)

            mlp_dense0_source_key = [
                f"{layer_name}/MlpBlock_3/Dense_0/{name}" for name in ["kernel", "bias"]]
            mlp_dense1_source_key = [
                f"{layer_name}/MlpBlock_3/Dense_1/{name}" for name in ["kernel", "bias"]]
            target_encoder.pwffn.dense_1.set_weights(
                [params[key] for key in mlp_dense0_source_key])
            target_encoder.pwffn.dense_2.set_weights(
                [params[key] for key in mlp_dense1_source_key])

            source_used_keys.extend(
                mlp_dense0_source_key + mlp_dense1_source_key)

            query_source_key = [
                f"{layer_name}/MultiHeadDotProductAttention_1/query/{name}" for name in ["kernel", "bias"]]
            key_source_key = [
                f"{layer_name}/MultiHeadDotProductAttention_1/key/{name}" for name in ["kernel", "bias"]]
            value_source_key = [
                f"{layer_name}/MultiHeadDotProductAttention_1/value/{name}" for name in ["kernel", "bias"]]
            out_source_key = [
                f"{layer_name}/MultiHeadDotProductAttention_1/out/{name}" for name in ["kernel", "bias"]]

            q_kernel, q_bias = reshape_mha_matrix(params, query_source_key)
            k_kernel, k_bias = reshape_mha_matrix(params, key_source_key)
            v_kernel, v_bias = reshape_mha_matrix(params, value_source_key)
            out_kernel, out_bias = reshape_mha_out_matrix(
                params, out_source_key)

            target_encoder.mha.wQ.set_weights([q_kernel, q_bias])
            target_encoder.mha.wK.set_weights([k_kernel, k_bias])
            target_encoder.mha.wV.set_weights([v_kernel, v_bias])
            target_encoder.mha.FFN.set_weights([out_kernel, out_bias])

            source_used_keys.extend(
                query_source_key + key_source_key + value_source_key + out_source_key)

        transformer_encodernorm_weight_source_keys = [
            f"Transformer/encoder_norm/{name}" for name in ["scale", "bias"]]
        self.layernorm.set_weights([params[k]
                                   for k in transformer_encodernorm_weight_source_keys])

        source_used_keys.extend(transformer_encodernorm_weight_source_keys)

        head_source_keys = ["head/kernel", "head/bias"]

        if pretrained_top:
            assert self.num_classes == 1000, \
                "Given 'pretrained_top=True' but num_classes!=1000. ImageNet1k has 1000 classes."
            self.dense_out.set_weights([params[key]
                                       for key in head_source_keys])
            source_used_keys.extend(head_source_keys)

        unused_keys = set(source_used_keys) ^ set(all_source_keys)

        if len(unused_keys) == 0:
            print(f"Weights loaded for {self.vit_size}")
        elif list(unused_keys).sort() == head_source_keys.sort() and not pretrained_top:
            print(f"Did not load the last top layer as pretrained_top=False")
        else:
            for unused_key in unused_keys:
                print(f"Couldn't load {unused_key}")
