# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np

BN_EPSILON = 0.001
decay = 0.9

def BN(input, conf):
    with tf.name_scope('BN'):
        params_shape = input.shape.as_list()[-1]
        axis = list(range(len(input.shape.as_list())-1)) # 得到需要计算batch的部分，除了最后一个维度不进行
        shift = tf.Variable(np.zeros(params_shape).astype('float32'), name='beta')
        scale = tf.Variable(np.ones(params_shape).astype('float32'), name='gamma')

        # 下面是对于非训练时使用的，相当于不使用BN操作，但是进行一个线性变换 gamma*input+beta, 所以如果有了BN，就没必要加上bias
        #  没有加下面的操作的时候，不能够用于NN层的测试，因为当NN层仅为1batch的时候，会全部变成0
        moving_mean = tf.Variable(np.zeros(params_shape).astype('float32'), trainable=False, name='moving_mean')
        moving_variance = tf.Variable(np.ones(params_shape).astype('float32'), trainable=False, name='moving_mean')
        # 下面是相当于对训练过程中使用的BN操作，需要计算平均值与方差

        batch_mean, batch_var = tf.nn.moments(input, axis)

        train_mean = tf.assign(moving_mean, moving_mean*decay + batch_mean*(1-decay))
        train_var = tf.assign(moving_variance, moving_variance*decay+batch_var*(1-decay))
        # if a value has no connection with the loss, and you dnt fetches it, it will not change
        mean, var = tf.cond(conf['is_training'],
                            lambda: (batch_mean, batch_var),
                            lambda: (moving_mean, moving_variance))

        tf.add_to_collection('moving_mean', moving_mean)
        tf.add_to_collection('moving_var', moving_variance)
        tf.add_to_collection('mean', mean)
        tf.add_to_collection('var', var)
        tf.add_to_collection('ops', train_mean)
        tf.add_to_collection('ops', train_var)

        result = tf.nn.batch_normalization(x=input, mean=mean, variance=var, offset=shift, scale=scale,
                                           variance_epsilon=BN_EPSILON)
    return result

