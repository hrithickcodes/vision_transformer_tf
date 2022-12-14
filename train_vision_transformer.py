import os
from vit import viT
import numpy as np
import tensorflow as tf
from utils import *

# we do not want tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# loading the config file
config = load_config(CONFIG_FILEPATH="vit_config.yaml")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = y_train.flatten().astype('float32')
y_test = y_test.flatten().astype('float32')

num_classes = len(np.unique(y_train))

training_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))


# zipping all the input variables and output variables into tf.data.Datasets object
train_ds = tf.data.Dataset.zip(training_dataset)\
    .cache()\
    .batch(batch_size=config["training_settings"]["batch_size"])\
    .shuffle(y_train.shape[0])\
    .prefetch(tf.data.experimental.AUTOTUNE)
train_ds = train_ds.map(lambda image, label: (preprocess_image(image), label))

    
test_ds = tf.data.Dataset.zip(test_dataset)\
    .cache()\
    .batch(batch_size=config["training_settings"]["batch_size"])\
    .shuffle(y_test.shape[0])\
    .prefetch(tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(lambda image, label: (preprocess_image(image), label))\

print("All data loaded and preprocessed...")

# building the viT model
print("Building model...")
model = viT(image_size=config["model_architecture"]["image_size"],
            patch_size=config["model_architecture"]["patch_size"],
            mlp_units=config["model_architecture"]["mlp_units"],
            patch_embedding_dim=config["model_architecture"]["patch_embedding_dim"],
            dropout_rate=config["model_architecture"]["dropout_rate"],
            num_stacked_encoders=config["model_architecture"]["num_transformer_encoder"],
            num_attention_heads=config["model_architecture"]["num_attention_heads"],
            num_classes=num_classes,
            name=config["model_architecture"]["model_name"])
print("Model built...")

# warming up the model
dummy_input = tf.random.normal(
    shape=[1] + config["model_architecture"]["image_size"])
_ = model(dummy_input)

print(os.linesep)
model.summary()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=config["training_settings"]["learning_rate"])

# Iterate over epochs.
print(os.linesep)
print("Starting training...")
print(os.linesep)

steps_per_epoch = len(train_ds) 
epoch_info = {
    "Training Accuracy": [],
    "Test Accuracy": [],
    "Training Loss": [],
    "Test Loss": []
}

max_test_accuracy = 0
for epoch in range(config["training_settings"]["epochs"]):
    batch_info = {
        "train_accs": [],
        "test_accs": [],
        "test_loss": [],
        "train_losses": [],
    }
    for step, (x_train, y_train) in enumerate(train_ds):
        batch_train_loss, batch_predictions = training_step(
            model, optimizer, x_train, y_train)
        batch_acc = compute_accuracy(y_train, batch_predictions)
        batch_info["train_losses"].append(batch_train_loss)
        batch_info["train_accs"].append(batch_acc)
        print(f"Batch [{step + 1}|{steps_per_epoch}] Batch loss: {batch_train_loss}, batch accuracy: {batch_acc}")
    print(os.linesep)

    for step, (x_test, y_test) in enumerate(test_ds):
        test_logits = model(x_test)
        test_loss = compute_loss(y_test, test_logits)
        test_acc = compute_accuracy(y_test, test_logits)
        batch_info["test_accs"].append(test_acc)
        batch_info["test_loss"].append(test_loss)

    epoch_train_acc = np.median(batch_info["train_accs"])
    epoch_train_loss = np.median(batch_info["train_losses"])
    epoch_test_acc = np.median(batch_info["test_accs"])
    epoch_test_loss = np.median(batch_info["test_loss"])

    epoch_info["Training Loss"].append(epoch_train_loss)
    epoch_info["Test Loss"].append(epoch_test_loss)
    epoch_info["Training Accuracy"].append(epoch_train_acc)
    epoch_info["Test Accuracy"].append(epoch_test_acc)
    


    print(f"Epoch: {epoch + 1}, Training accuracy: {epoch_train_acc}, Training Loss: {epoch_train_loss}, Test Loss: {epoch_test_loss}, Test accuracy: {epoch_test_acc}")
    if epoch_test_acc > max_test_accuracy:
        model_saving_path = os.path.join(config["save_paths"]["model_save_path"],
                                         config["model_architecture"]["model_name"])
        model.save_weights(model_saving_path, save_format='tf')
        print(f"Model saved. Accuracy Improved : {epoch_test_acc - max_test_accuracy}")
        max_test_accuracy = epoch_test_acc
    print(os.linesep)
    

if config["training_settings"]["save_plots"]:
    plot_accuracy(epoch_info,
                  savedir=config["save_paths"]["plot_save_path"],
                  show=config["training_settings"]["show_plots"])
    plot_loss(epoch_info,
              savedir=config["save_paths"]["plot_save_path"],
              show=config["training_settings"]["show_plots"])
