
# Vision Transformer implementation in TensorFlow 2.x

This repository is the unofficial implementation of the paper named 
**AN IMAGE IS WORTH 16X16 WORDS:
TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE**. While the Transformer architecture has become the de-facto standard for natural
language processing tasks, this paper used the transformer architecture and achived SOTA performance in image recognition.

<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Vision_Transformer.gif/800px-Vision_Transformer.gif" width="450" height = "290">
</div>

## Setup
The repository can be cloned using the below commands.


```
git clone https://github.com/TheTensorDude/vision_transformer_tf.git
cd vision_transformer_tf
pip install -r requirements.txt
```


### Training vision transformer :metal:
Before training the hyperparameters need to be set in the vit_config.yaml file.

```yaml 
model_architecture:
    image_size: [32, 32, 3]
    patch_size: 2
    patch_embedding_dim: 512
    num_attention_heads: 8
    num_transformer_encoder: 6
    mlp_units: 512
    dropout_rate: 0.1
    model_name: cifar_viT


training_settings:
    epochs: 120
    batch_size: 8
    learning_rate: 0.0002 
    save_plots: True
    show_plots: False
    
save_paths:
    model_save_path: model_weights
    plot_save_path: runs
```
After configuring the YAML file the next step is to run the following command to start training the model. Note that, the model will start training on CIFAR-10 dataset by default.
```
python train_vision_transformer.py
```

Output 

```
Epoch: 6, Training accuracy: 0.65625, Training Loss: 0.9600569009780884, Test Loss: 1.1582378149032593, Test accuracy: 0.59375


Batch [1|782] Batch loss: 0.899474561214447, batch accuracy: 0.640625
Batch [2|782] Batch loss: 0.6949890851974487, batch accuracy: 0.75
Batch [3|782] Batch loss: 0.774124026298523, batch accuracy: 0.734375
Batch [4|782] Batch loss: 0.8040367364883423, batch accuracy: 0.765625
Batch [5|782] Batch loss: 0.6990523338317871, batch accuracy: 0.78125
```

