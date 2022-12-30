import json
import yaml, os
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

def load_config(CONFIG_FILEPATH):
    with open(CONFIG_FILEPATH, 'r') as config:
        config = yaml.safe_load(config)
        return config

def load_imagenet_classes(filepath):
    with open(filepath,'r') as imgnet_file:
        classes_dict = eval(imgnet_file.read())
    return classes_dict

def preprocess(input_tensor):
    input_tensor = tf.keras.applications.imagenet_utils.preprocess_input(input_tensor,
                                                               data_format=None,
                                                               mode="tf")
    return input_tensor

def training_step(model, optimizer, input_variables, output_variables):
    with tf.GradientTape() as tape:
        logits = model(input_variables, training = True)
        vit_loss = compute_loss(output_variables, logits)
    grads = tape.gradient(vit_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return vit_loss.numpy(), logits
    
def predict(model, query_image):
    logits = model(query_image, training = False)
    normalised_logits = tf.nn.softmax(logits, axis = -1)
    predictions = tf.argmax(normalised_logits, axis = -1)
    return predictions
    
    