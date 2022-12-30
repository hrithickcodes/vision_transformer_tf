import os
import argparse
from vit import viT
import tensorflow as tf 
from utils.general import load_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--vit-size',
                        type=str,
                        required = True,
                        help='The vit model that wa finetuned')
    
    parser.add_argument('--num-classes',
                    type=int,
                    required = True,
                    help='Number of classes on which it was train/finetuned on')
    
    parser.add_argument('--source-name',
                    type=str,
                    required = True,
                    help='Suffix of the finetuned model')
    
    parser.add_argument('--tflite-save-name',
                type=str,
                default = "vit.tflite",
                help='name of the tflite model for saving')

    return parser.parse_args()

args = parse_opt()

VIT_CONFIG = load_config("vit_architectures.yaml")


model = viT(args.vit_size, args.num_classes)
model.load_weights(os.path.join("finetuning_weights", args.source_name)).expect_partial()
model.compute_output_shape(input_shape = [1] + VIT_CONFIG[args.vit_size]["image_size"])

model.summary()
print(os.linesep)

print("Conversion started..")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
vit_tflite = converter.convert()
open(args.tflite_save_name, "wb").write(vit_tflite)


print(f"{args.tflite_save_name} saved.")