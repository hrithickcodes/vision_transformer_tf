import os
import numpy as np

def load_npz_file(pretrained_size):
    pretrained_model_name = f"{pretrained_size}_imagenet21k+imagenet2012.npz"
    relative_weights_path = os.path.join("pretrained_weights", pretrained_model_name)
    weights_dict = np.load(
        relative_weights_path,
        allow_pickle=False
    )
    return weights_dict

def get_pretrained_encoders_num(pretrained_param_dict):
    num_pretrained_encoders = len(
        set("/".join(layer.split("/")[:2]) for layer in
            list(pretrained_param_dict.keys())
            if layer.startswith("Transformer/encoderblock_"))
    )
    return num_pretrained_encoders

def reshape_mha_matrix(params, source_keys):
    kernel_key, bias_key = source_keys
    kernel_shape, bias_shape = params[kernel_key].shape, params[bias_key].shape
    reshaped_kernel = params[kernel_key].reshape(-1, kernel_shape[-1] * kernel_shape[-2])
    reshaped_bias = params[bias_key].reshape(bias_shape[0] * bias_shape[1])
    return reshaped_kernel, reshaped_bias


def reshape_mha_out_matrix(params, out_key):
    kernel_key, bias_key = out_key
    kernel_shape, _ = params[kernel_key].shape, params[bias_key].shape
    reshaped_kernel = params[kernel_key].reshape(kernel_shape[0] * kernel_shape[1], -1)
    return reshaped_kernel, params[bias_key]

