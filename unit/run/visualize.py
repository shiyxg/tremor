# -*- coding: utf-8 -*-

from __future__ import print_function

# Graph is built in Graph.py
import sys
from run.get_train_data import Data
# https://github.com/InFoCusp/tf_cnnvis
import tf_cnnvis
sys.path.append('/home/pkushi/tremor')

#path = 'C:\\Users\\shi\\oneDrive\\FaultsDetection\\fault-test'
data_train = Data('/home/pkushi/dataset')
data_test = Data('/home/pkushi/dataset_test')
NUM = 1
IS_TRAINING = True

from run.graph import *
def train_cnn():
    g = cnn_graph()
    '''
    g = {
        'images': images,
        'labels': labels_r,
        'learning_rate': learning_rate,
        'logits': logits,
        'loss': loss,
        'train': train,
        'summary': summary
    }
    '''
    images = g['images']
    labels = g['labels']
    learning_rate = g['learning_rate']
    train = g['train']
    loss = g['loss']
    logits = g['logits']
    is_training = g['is_training']
    with tf.Session() as sess:
        pwd = '/home/pkushi/CNNlog/CNN/BN_test'
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess,'/home/pkushi/CNNlog/CNN/tremor_with_norm_BN_test5/para_6000/')

        # saver.save(sess,'/home/pkushi/CNNlog/vis.c')

        mean = tf.get_collection('mean')[-1]
        var = tf.get_collection('var')[-1]
        BN_reuslt = tf.get_collection('BN_result')[-1]
        wx_result = tf.get_collection('wx_result')[-1]
        conv_result = tf.get_collection('conv_result')
        fetchVariables = [loss, logits, mean, var, BN_reuslt, wx_result, conv_result]

        for i in range(1):
            trainBatch = data_test.get_tremors(NUM)
            # train
            feedData = {images: trainBatch[0],
                        labels: trainBatch[1],
                        learning_rate: 1e-3,
                        is_training: False}
            tf_cnnvis.deconv_visualization(sess, feedData, input_tensor=None, layers='p',
                                           path_logdir='/home/pkushi/CNNlog/vis/log', path_outdir='/home/pkushi/CNNlog/vis/output')
            print(1)

    return sess

train_cnn()