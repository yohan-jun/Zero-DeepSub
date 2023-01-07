import tensorflow as tf
import numpy as np


def conv_layer_k(input_data, conv_filter, is_relu=False, is_scaling=False):
    """
    Parameters
    ----------
    x : input data
    conv_filter : weights of the filter
    is_relu : applies  ReLU activation function
    is_scaling : Scales the output

    """

    W = tf.get_variable('W_k', shape=conv_filter, initializer=tf.keras.initializers.he_normal())
    x = tf.nn.conv2d(input_data, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.keras.layers.BatchNormalization()(x)

    if (is_relu):
        x = tf.nn.leaky_relu(x, alpha=0.05)

    if (is_scaling):
        scalar = tf.constant(0.1, dtype=tf.float32)
        x = tf.multiply(scalar, x)

    return x


def conv_layer_i(input_data, conv_filter, is_relu=False, is_scaling=False):
    """
    Parameters
    ----------
    x : input data
    conv_filter : weights of the filter
    is_relu : applies  ReLU activation function
    is_scaling : Scales the output

    """

    W = tf.get_variable('W_i', shape=conv_filter, initializer=tf.keras.initializers.he_normal())

    x = tf.nn.conv2d(input_data, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.keras.layers.BatchNormalization()(x)

    if (is_relu):
        x = tf.nn.leaky_relu(x, alpha=0.05)

    if (is_scaling):
        scalar = tf.constant(0.1, dtype=tf.float32)
        x = tf.multiply(scalar, x)

    return x


def ResNet_K(input_data, nb_res_blocks, nb_basis):
    """

    Parameters
    ----------
    input_data : nrow x ncol x 2. Regularizer Input
    nb_res_blocks : default is 15.

    conv_filters : dictionary containing size of the convolutional filters applied in the ResNet
    intermediate outputs : dictionary containing intermediate outputs of the ResNet

    Returns
    -------
    nw_output : nrow x ncol x 2 . Regularizer output

    """
    num_chan = 128

    conv_filters = dict([('w1_k', (3, 3, nb_basis * 2, num_chan)), ('w2_k', (3, 3, num_chan, num_chan)), ('w3_k', (3, 3, num_chan, nb_basis * 2))])
    intermediate_outputs = {}

    with tf.variable_scope('FirstLayer'):
        intermediate_outputs['layer0'] = conv_layer_k(input_data, conv_filters['w1_k'], is_relu=True, is_scaling=False)

    for i in np.arange(1, nb_res_blocks + 1):
        with tf.variable_scope('ResBlock' + str(i)):
            intermediate_outputs['layer' + str(i)] = conv_layer_k(intermediate_outputs['layer' + str(i - 1)], conv_filters['w2_k'], is_relu=True, is_scaling=False)

    with tf.variable_scope('LastLayer'):
        rb_output = conv_layer_k(intermediate_outputs['layer' + str(i)], conv_filters['w3_k'], is_relu=False, is_scaling=False)

    with tf.variable_scope('Residual'):
        nw_output = rb_output

    return nw_output


def ResNet_I(input_data, nb_res_blocks, nb_basis):
    """

    Parameters
    ----------
    input_data : nrow x ncol x 2. Regularizer Input
    nb_res_blocks : default is 15.

    conv_filters : dictionary containing size of the convolutional filters applied in the ResNet
    intermediate outputs : dictionary containing intermediate outputs of the ResNet

    Returns
    -------
    nw_output : nrow x ncol x 2 . Regularizer output

    """
    num_chan = 128

    conv_filters = dict([('w1_i', (3, 3, nb_basis * 2, num_chan)), ('w2_i', (3, 3, num_chan, num_chan)), ('w3_i', (3, 3, num_chan, nb_basis * 2))])
    intermediate_outputs = {}

    with tf.variable_scope('FirstLayer'):
        intermediate_outputs['layer0'] = conv_layer_i(input_data, conv_filters['w1_i'], is_relu=True, is_scaling=False)

    for i in np.arange(1, nb_res_blocks + 1):
        with tf.variable_scope('ResBlock' + str(i)):
            intermediate_outputs['layer' + str(i)] = conv_layer_i(intermediate_outputs['layer' + str(i - 1)], conv_filters['w2_i'], is_relu=True, is_scaling=False)

    with tf.variable_scope('LastLayer'):
        rb_output = conv_layer_i(intermediate_outputs['layer' + str(i)], conv_filters['w3_i'], is_relu=False, is_scaling=False)

    with tf.variable_scope('Residual'):
        nw_output = rb_output

    return nw_output


def ResNet(input_data, nb_res_blocks, nb_basis):
    """

    Parameters
    ----------
    input_data : nrow x ncol x 2. Regularizer Input
    nb_res_blocks : default is 15.

    conv_filters : dictionary containing size of the convolutional filters applied in the ResNet
    intermediate outputs : dictionary containing intermediate outputs of the ResNet

    Returns
    -------
    nw_output : nrow x ncol x 2 . Regularizer output

    """

    conv_filters = dict([('w1_i', (3, 3, nb_basis * 2, 128)), ('w2_i', (3, 3, 128, 128)), ('w3_i', (3, 3, 128, nb_basis * 2))])
    intermediate_outputs = {}

    with tf.variable_scope('FirstLayer'):
        intermediate_outputs['layer0'] = conv_layer_i(input_data, conv_filters['w1_i'], is_relu=True, is_scaling=False)

    for i in np.arange(1, nb_res_blocks + 1):
        with tf.variable_scope('ResBlock' + str(i)):
            intermediate_outputs['layer' + str(i)] = conv_layer_i(intermediate_outputs['layer' + str(i - 1)], conv_filters['w2_i'], is_relu=True, is_scaling=False)

    with tf.variable_scope('LastLayer'):
        rb_output = conv_layer_i(intermediate_outputs['layer' + str(i)], conv_filters['w3_i'], is_relu=False, is_scaling=False)

    with tf.variable_scope('Residual'):
        nw_output = rb_output

    return nw_output


def mu_param():
    """
    Penalty parameter used in DC units, x = (E^h E + \mu I)^-1 (E^h y + \mu * z)
    """

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        mu = tf.get_variable(name='mu', dtype=tf.float32, initializer=.005, trainable=True)

    return mu
