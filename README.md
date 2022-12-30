# Vision Transformer in TensorFlow 2.x 🚀

This repository contains code for the paper named [AN IMAGE IS WORTH 16X16 WORDS:
TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929v1.pdf). This paper proves reliance on CNNs is not necessary, and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks, when pre-trained on large amounts of data.

<div align="center">
<a align="center" href="https://arxiv.org/pdf/2010.11929v1.pdf" target="_blank">
<img width="800" src="images\model.jpg"></a>
</div>

### Setup
Use the following commands to install all the necessary packages. A CUDA-enabled GPU is necessary for faster training and inference.

```
git clone https://github.com/TheTensorDude/vision_transformer_tf.git
cd vision_transformer_tf
pip install -r requirements.txt
```

### Download pretrained weights
Vision transformers were first trained on ImageNet21k and then it was finetuned on ImageNet1k. The weights are in .npz format and can be downloaded from this [gdrive folder](https://drive.google.com/drive/folders/110vr3yQb-_9e-b37DbRmyo07pa7Z_88h?usp=share_link) and put them inside the **pretrained_weights** folder.

### Usage
Once the setup is done, the pre-trained model can be loaded with just two lines of code. The available vit models sizes are **ViT-BASE16, ViT-BASE32, ViT-LARGE16, ViT-LARGE32**.

```python
from vit import viT
vit_large = viT(vit_size="ViT-LARGE32")
vit_large.from_pretrained(pretrained_top=True)
```

Inference using ImageNet1k labels can be done using the following snippet of code.

```python
import os
from vit import viT
import tensorflow as tf
from utils.general import load_imagenet_classes

# loading imageNet1k classes
classes = load_imagenet_classes(filepath=os.path.join("pretrained_weights","imagenet_2012.txt"))

# intializing vit
vit_large = viT(vit_size="ViT-BASE32")

# loading pretrained weights
vit_large.from_pretrained(pretrained_top=True)

# loading and decoding the image
image = tf.image.decode_jpeg(tf.io.read_file('goldfish.jpg'))

# necessary imagenet preprocessing
preprocessed_image = tf.expand_dims(vit_large.preprocess_input(image), axis = 0)

# prediction using pretrained model
output = vit_large.predict(preprocessed_image)

print(f"Predicted class: {classes[output.argmax()]}")

# Output: goldfish, Carassius auratus
```


### Finetuning on custom dataset

The first step is to download the dataset into the datasets folder, then the finetuning can be started using the below command.

```
# finetuning the BASE16 size for 2 epochs on a custom dataset.
python train.py \
    --training-data dataset/training_set --test-data dataset/test_set \
    --num-classes 2 \
    --epochs 2 \
    --batch-size 16 \
    --vit-size ViT-BASE16 \
    --model-name ViT-BASE16_cat_dog \
    --save-training-stats 
```
output when finetuned to classify cat and dog.
```
Did not load the last top layer as pretrained_top=False

Found 8005 files belonging to 2 classes.
Found 2023 files belonging to 2 classes.


Epoch 1/2
501/501 [==============================] - 1316s 3s/step - loss: 0.0512 - acc: 0.9763 - val_loss: 0.0153 - val_acc: 0.9970
Epoch 2/2
501/501 [==============================] - 1147s 2s/step - loss: 0.0037 - acc: 0.9989 - val_loss: 0.0150 - val_acc: 0.9975


Accuracy results saved at runs/train-test-accuracy.png
Loss results saved at runs/train-test-loss.png
```

### Converting to tflite

Every vit model can be converted to tflite using this below command.

```
 python export_to_tflite.py --vit-size ViT-BASE16 \
        --source-name ViT-BASE16_cat_dog \
        --num-classes 2 \
        --tflite-save-name cat_dog.tflite 
```

### Tasks
- [ ] Adding a tutorial.
- [ ] support for ImageNet21k weights.
- [ ] Cosine decay learning rate for finetuning.
- [ ] Add support for ViT-BASE8 and ViT-Hybrid.
