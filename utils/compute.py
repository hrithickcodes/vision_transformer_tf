import tensorflow as tf

def compute_loss(labels, logits):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss = scce(labels, logits)
    return tf.reduce_mean(loss)

def compute_accuracy(labels, logits):
    normalised_logits = tf.nn.softmax(logits, axis = -1)
    predictions = tf.argmax(normalised_logits, axis = -1)
    acc = tf.keras.metrics.Accuracy()
    return acc(labels, predictions).numpy()

