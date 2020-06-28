import tensorflow as tf


def conv_layer_with_batch_normalization(
    inputs,
    filters,
    kernel_size,
    strides,
    padding,
    activation,
    training
):
    with tf.name_scope('BN_conv'):
        conv = tf.layers.conv2d(
            inputs, filters, kernel_size, strides, padding, use_bias=False)
        batch_norm = tf.layers.batch_normalization(conv, training=training)
        if activation is None:
            return batch_norm
        return activation(batch_norm)
