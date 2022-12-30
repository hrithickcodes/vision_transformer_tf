import tensorflow as tf

def vit_loss(y_true, y_pred):
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    y_true = tf.expand_dims(y_true, axis = 0)
    _loss = scce(y_true, y_pred)
    return tf.reduce_mean(_loss)