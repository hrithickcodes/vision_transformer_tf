import os
import json
from vit import viT
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

weight_path = r"finetuning_weights\ViT-BASE16_cat_dog"
class_indices_path = r"finetuning_weights\class_indices.json"

with open(class_indices_path, 'r') as fp:
    class_indices = json.load(fp)

model = viT(vit_size="ViT-BASE16",
            num_classes=2,
            class_activation="softmax")

model.load_weights(weight_path)

image = tf.image.decode_jpeg(tf.io.read_file('test.jpg'))
preprocessed_image = model.preprocess_input(image)
model_input = tf.expand_dims(preprocessed_image, axis=0)

output = model.predict(model_input)
prediction = tf.argmax(output, axis=-1).numpy()[0]
print(f"Predicted class:", class_indices[str(prediction)])
