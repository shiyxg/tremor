import tensorflow as tf


def max_pool(input, name='pool', conf={}):

    if conf.get('ksize') is None:
        ksize=[1, 2, 2, 1]
    else:
        ksize=conf['ksize']

    if conf.get('stride') is None:
        stride=[1, 2, 2, 1]
    else:
        stride=conf['stride']

    with tf.name_scope(name):
        output = tf.nn.max_pool(input, ksize=ksize, strides=stride, padding='SAME')
    return output


def max_pool_1d(input, name='pool', conf={}):

    if conf.get('ksize') is None:
        ksize=[4]
    else:
        ksize = [conf['ksize'][1]]

    if conf.get('stride') is None:
        stride=[4]
    else:
        stride=[conf['stride'][1]]

    with tf.name_scope(name):
        output = tf.nn.pool(input, window_shape=ksize, pooling_type='MAX', strides=stride, padding='SAME')
    return output