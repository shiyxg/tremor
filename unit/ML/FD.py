# -*- coding: utf-8 -*-

from __future__ import print_function

# Graph is built in Graph.py
import tensorflow as tf
import numpy as np
import nibabel as nb
import sys
import scipy.io as io
import os
import matplotlib as mpl
mpl.use('agg')
import matplotlib.cm as cm
import matplotlib.pyplot as pyplot

#path = 'C:\\Users\\shi\\oneDrive\\FaultsDetection\\fault-test'

path = '/gpfs/share/home/1400012437/FaultDetection/fault-test'
sys.path.append(path)
sys.path.append('/gpfs/share/home/1400012437/FaultDetection/fault-test/FCN')

from .dataAnalysis.nii import *
from .graph import *
from .path import *


def saveSummary(step, data, sess, shape, trainWrite, testWrite, g):
    images = g['images']
    labels = g['labels']
    learning_rate = g['learning_rate']
    summary = g['summary']

    trainBatch = data.trainBatchLayer(1, shape=shape)
    testBatch = data.testBatchLayer(1, shape=shape)

    trainFeed = {images: trainBatch[0].reshape([1,250,400,1]),
                 labels: trainBatch[1].reshape([1,250,400,1]),
                 learning_rate: 1e-4}
    testFeed = {images: testBatch[0].reshape([1,250,400,1]),
                labels: testBatch[1].reshape([1,250,400,1]),
                learning_rate: 1e-4}

    fetchVariables = [summary]

    [summary_train
     ] = sess.run(fetches=fetchVariables, feed_dict=trainFeed)
    [summary_test
     ] = sess.run(fetches=fetchVariables, feed_dict=testFeed)

    trainWrite.add_summary(summary_train, step)
    testWrite.add_summary(summary_test, step)
    trainWrite.flush()
    testWrite.flush()

    LR_i = 1e-3

    return LR_i


def train(trainPath,testPath,validPath):
    g = fcn_graph()
    # Value = {
    #     'images': images,
    #     'labels': labels,
    #     'learning_rate': learning_rate,
    #     'logits': logits,
    #     'loss': loss,
    #     'train': train_step
    # }
    images = g['images']
    labels = g['labels']
    learning_rate = g['learning_rate']
    train = g['train']

    data = niiS(trainPath, testPath, expand=0)
    with tf.Session() as sess:
        pwd = '/gpfs/share/home/1400012437/CNNlog/FCN/PKU/04'
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        # saver.restore(sess,'/home/shi/CNNlog/FCN/FD/fullLayer07/para/')

        command = ['mkdir ' + pwd,
                   'rm -rf ' + pwd + '/*',
                   'mkdir ' + pwd + '/code',
                   'mkdir ' + pwd + '/sample',
                   'cp -r /gpfs/share/home/1400012437/FaultDetection/fault-test/* ' + pwd + '/code']

        for i in command:
            os.system(i)

        trainWrite = tf.summary.FileWriter(pwd + '/train', tf.get_default_graph())
        testWrite = tf.summary.FileWriter(pwd + '/test', tf.get_default_graph())

        for i in range(50001):
            trainBatch = data.trainBatchLayer(10, shape=[250,400])
            # train
            feedData = {images: trainBatch[0].reshape([1,250,400,1]),
                        labels: trainBatch[1].reshape([1,250,400,1]),
                        learning_rate: 1e-3}

            fetchVariables = [train]

            [_] = sess.run(fetches=fetchVariables, feed_dict=feedData)

            if i % 100 == 0:
                saveSummary(i, data, sess, [250,400], trainWrite, testWrite, g)
                print(i)
            if i % 15000 == 0 and i != 0:
                saver.save(sess, pwd + '/para_%g/' % i)
                print(1)
        saver.save(sess, pwd + '/para/')
        trainWrite.close()
        testWrite.close()

    return sess

train(trainPath, testPath,validPath)