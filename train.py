import os
import json
import argparse
from vit import viT
import tensorflow as tf
from utils.loss import vit_loss
from utils.plots import plot_accuracy, plot_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help='Total training epochs for finetuning')

    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='total batch size for GPUs')

    parser.add_argument('--training-data',
                        type=str,
                        help='path to the training data')

    parser.add_argument('--test-data',
                        type=str,
                        help='path to the test data')

    parser.add_argument('--vit-size',
                        type=str,
                        default="ViT-BASE16",
                        help='The size of the vit model to finetune.')

    parser.add_argument('--num-classes',
                        type=int,
                        help='Number of classes to finetune on.')

    parser.add_argument('--vit-config',
                        type=str,
                        default="vit_architectures.yaml",
                        help='architectures for vit models')

    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='learning-rate to use for finetuning the model.')

    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='learning-rate to use for finetuning the model.')

    parser.add_argument('--validation-batch-size',
                        type=int,
                        default=16,
                        help='validation batch size for calculating accuracy.')

    parser.add_argument('--global-clipnorm',
                        type=float,
                        default=1.0,
                        help='The paper uses a global clipnorm of 1 while finetuning')

    parser.add_argument('--model-name',
                        type=str,
                        default="saved_model",
                        help='Finetuned model name.')
    
    parser.add_argument('--pretrained-top', action='store_true')
    parser.add_argument('--save-training-stats', action='store_true')
    parser.add_argument('--train-from-scratch', action='store_false')

    return parser.parse_args()


args = parse_opt()

vit = viT(vit_size=args.vit_size,
          num_classes=args.num_classes,
          config_path=args.vit_config)

if args.train_from_scratch:
    vit.from_pretrained(pretrained_top=args.pretrained_top)

print(os.linesep)

train_ds = tf.keras.utils.image_dataset_from_directory(
    args.training_data,
    image_size=(vit.image_height, vit.image_width),
    batch_size=args.batch_size)

with open(os.path.join("finetuning_weights", "class_indices.json"), 'w') as fp:
    class_indices = {i: k for i, k in enumerate(train_ds.class_names)}
    json.dump(class_indices,
              fp,
              indent=4)

train_ds = train_ds.map(lambda image, label: (
    vit.preprocess_input(image), label))

test_ds = tf.keras.utils.image_dataset_from_directory(
    args.test_data,
    image_size=(vit.image_height, vit.image_width),
    batch_size=args.batch_size)\
    .map(lambda image, label: (vit.preprocess_input(image), label))


print(os.linesep)

optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate,
                                    momentum=args.momentum,
                                    global_clipnorm=args.global_clipnorm)

chekpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join("finetuning_weights", f"{args.vit_size}_{args.model_name}"),
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=True)

vit.compile(optimizer=optimizer,
            loss=vit_loss,
            metrics=["acc"])

history = vit.fit(train_ds,
                  validation_data=test_ds,
                  shuffle=True,
                  validation_batch_size=args.validation_batch_size,
                  callbacks=[chekpoint],
                  epochs=args.epochs)

print(os.linesep)

if args.save_training_stats:
    plot_accuracy(history, "runs")
    plot_loss(history, "runs")