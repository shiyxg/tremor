# -*- coding = utf-8 -*-
from __future__ import print_function

import tensorflow as tf
import numpy as np
from unit.BN import BN
# 下面是一些与卷积有关的操作，包括二维卷积，三维卷积等，或者构建2,3维卷积层


def conv2d(input, conf):
    # conf need keys: filterSize, strideSize, outputChn,  and optional filterStddev
    # Setting the conv core's shape
    inputChn = input.get_shape().as_list()[-1]
    if len(conf['filterSize']) == 2:
        coreSize_1, coreSize_2 = conf['filterSize']
    elif len(conf['filterSize']) == 1:
        coreSize_1, coreSize_2 = [conf['filterSize'], conf['filterSize']]
    else:
        raise ValueError('conf["filterSize"] Error')
    outputChn = conf['outputChn']
    coreShape = [coreSize_1, coreSize_2, inputChn,outputChn]

    # get the stride
    if len(conf['strideSize']) == 2:
        strideSize_1, strideSize_2 = conf['strideSize']
    elif len(conf['strideSize']) == 1:
        strideSize_1, strideSize_2 = [conf['strideSize'],conf['strideSize']]
    else:
        strideSize_1, strideSize_2 = [1, 1]
    stride=[1, strideSize_1, strideSize_2, 1]

    if conf.get('filterStddev') is None:
        # Use Glorot and Bengio(2010)'s init method
        n_in = inputChn*coreSize_1*coreSize_2
        n_out = outputChn*coreSize_2*coreSize_1*1.0/(strideSize_1*strideSize_2)
        stddev = np.sqrt(2.0/(n_in+n_out))
        # 高斯分布
        core_init = tf.truncated_normal(mean=0, stddev=stddev, shape=coreShape)
        # 均匀分布
        # core_init = np.random.uniform(low =-np.sqrt(3)*stddev,
        #                               high= np.sqrt(3)*stddev,
        #                               size= coreShape).astype('float32')
    else:
        core_init = np.random.uniform(low=-np.sqrt(3) * conf['filterStddev'],
                                      high=np.sqrt(3) * conf['filterStddev'],
                                      size=coreShape).astype('float32')

    if conf.get('padding') is not None:
        padding = conf.get('padding')
    else:
        padding = 'SAME'

    with tf.name_scope('conv2d'):
        core = tf.Variable(core_init,name='convCore')
        result = tf.nn.conv2d(input, core, strides=stride, padding=padding, data_format='NHWC')
        '''
        if use_bias:
            bias_init = np.zeros(outputChn).astype('float32')
            bias = tf.Variable(bias_init,name='bias')
            result = tf.nn.bias_add(result,bias)

        if act is not None:
            result = act(result)
        '''
    return result


def conv2d_layer(input, conf):
    '''
    :param input: 卷积层的输入，要求格式NHWC
    :param conf: 卷积层有关的参数，包括：
                层名字：name
                二维卷积要用到的：filterSize， outputChn, 可选的：strideSize，与卷积核初始化的标准差filterStddev
                BN操作要用到的：is_training
    :return:经过卷积层之后的值
    '''

    with tf.name_scope(conf['name']):
        result = conv2d(input,conf)
        result = BN(result, conf)
        result = tf.nn.relu(result)
        tf.add_to_collection('conv_result', result)

    return result


def conv2d_block(input, conf):
    '''
    构建一个多个卷积层组合的块，所有卷积层参数一致
    :param input:卷积块的输入，要求格式NHWC
    :param conf:卷积层有关的参数，包括：
                这个块的名字：name
                二维卷积要用到的：filterSize， outputChn, 可选的：strideSize，与卷积核初始化的标准差filterStddev
                BN操作要用到的：is_training
                times:卷击块内部的卷积层的个数
    :return:返回值
    '''

    conf_conv = conf.copy()

    with tf.name_scope(conf['name']):
        for i in range(conf['times']):
            conf_conv['name'] = 'conv'+str(i+1)
            input = conv2d_layer(input, conf_conv)
    return input


def conv1d(input, conf):
    '''
    :param input: the input of conv1d
    :param conf:  some configure about this layer
    :return: the result of conv1d
    '''

    inputChn = input.get_shape().as_list()[-1]
    if len(conf['filterSize']) == 1:
        core_size = conf['filterSize'][0]
    else:
        raise ValueError('conf["filterSize"] Error')
    outputChn = conf['outputChn']
    core_shape = [core_size, inputChn, outputChn]

    if len(conf['strideSize']) == 1:
        stride_size = conf['strideSize'][0]
    else:
        stride_size = 1

    if conf.get('filterStddev') is None:
        # Use Glorot and Bengio(2010)'s init method
        n_in = inputChn*core_size
        n_out= outputChn*core_size*1.0/stride_size
        stddev = np.sqrt(2.0/(n_in+n_out))
        # 高斯分布
        core_init = tf.truncated_normal(mean=0, stddev=stddev, shape=core_shape)
        # 均匀分布
        # core_init = np.random.uniform(low =-np.sqrt(3)*stddev,
        #                               high= np.sqrt(3)*stddev,
        #                               size= coreShape).astype('float32')
    else:
        core_init = np.random.uniform(low=-np.sqrt(3) * conf['filterStddev'],
                                      high=np.sqrt(3) * conf['filterStddev'],
                                      size=core_shape).astype('float32')
    with tf.name_scope('conv1d'):
        core = tf.Variable(core_init, name='convCore')
        result = tf.nn.conv1d(input, core, stride=stride_size,padding='SAME',data_format='NWC')
    return result


def conv1d_layer(input, conf):
    with tf.name_scope(conf['name']):
        result = conv1d(input,conf)
        result = BN(result, conf)
        result = tf.nn.relu(result)
    return result


def conv1d_block(input, conf):
    conf_conv = conf.copy()

    with tf.name_scope(conf['name']):
        for i in range(conf['times']):
            conf_conv['name'] = 'conv' + str(i + 1)
            input = conv1d_layer(input, conf_conv)
    return input


conf_sample_2d = {
    'name'              :'conv',
    'times'             :2,
    'filterSize'        :[2,2],
    'outputChn'         :256,
    'strideSize'        :[1,1],
    'is_training'       :True
}

conf_sample_1d = {
    'name'              :'conv',
    'times'             :2,
    'filterSize'        :[2],
    'outputChn'         :256,
    'strideSize'        :[1],
    'is_training'       :True
}