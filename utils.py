import yaml, os
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt


def load_config(CONFIG_FILEPATH):
    with open(CONFIG_FILEPATH, 'r') as config:
        config = yaml.safe_load(config)
        return config
    
def preprocess_image(input):
    return tf.keras.applications.imagenet_utils.preprocess_input(input, data_format=None,mode="tf")



def compute_loss(labels, logits):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss = scce(labels, logits)
    return tf.reduce_mean(loss)

def compute_accuracy(labels, logits):
    normalised_logits = tf.nn.softmax(logits, axis = -1)
    predictions = tf.argmax(normalised_logits, axis = -1)
    acc = tf.keras.metrics.Accuracy()
    return acc(labels, predictions).numpy()


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
    
    
def plot_accuracy(epochdict, savedir, show = False):
    training_accs, test_accs = epochdict["Training Accuracy"], epochdict["Test Accuracy"]
    plt.grid(True)
    plt.plot(np.arange(len(training_accs)),training_accs, color = "blue")
    plt.plot(np.arange(len(test_accs)), test_accs, color = "orange")
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.title('Accuracy vs Epoch')
    if savedir:
        savepath = os.path.join(savedir, "train-test-accuracy.png")
        plt.savefig(savepath)
    
    if show:    plt.show()
    plt.clf()
    print(f"Accuracy results saved at {savepath}")
    
    
    
def plot_loss(epochdict, savedir, show = True):
    training_losses, test_losses = epochdict["Training Loss"], epochdict["Test Loss"]
    
    plt.grid(True)
    plt.plot(np.arange(len(training_losses)),training_losses, color = "blue")
    plt.plot(np.arange(len(test_losses)), test_losses, color = "orange")
    plt.legend(['Training Loss', 'Test Loss'])
    plt.title('Loss vs Epoch')
    if savedir:
        savepath = os.path.join(savedir, "train-test-loss.png")
        plt.savefig(savepath)
    
    if show:    plt.show()
    plt.clf()
    print(f"Loss results saved at {savepath}")
    

    