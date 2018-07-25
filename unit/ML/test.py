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

from .dataAnalysis.nii import *
from .graph import *
from .path import *


def save_summary_fcn(step, data, sess, trainWrite, testWrite, g):
    images = g['images']
    labels = g['labels']
    learning_rate = g['learning_rate']
    summary = g['summary']

    trainBatch = data.trainBatchLayer(1, shape=[250, 400])
    # train
    feedData = {images: trainBatch[0].reshape([1, 250, 400, 1]),
                labels: trainBatch[1].reshape([1, 250, 400, 1]),
                learning_rate: 1e-3}
    fetchVariables = [summary]
    [summary_train] = sess.run(fetches=fetchVariables, feed_dict=feedData)
    print(1)
    testBatch = data.testBatchLayer(1)
    testBatch = [testBatch[0].reshape([1, 250, 400, 1]), testBatch[1].reshape([1, 250, 400, 1])]
    testFeed = {images: testBatch[0],
                labels: testBatch[1],
                learning_rate: 1e-3}
    [summary_test] = sess.run(fetches=fetchVariables, feed_dict=testFeed)

    trainWrite.add_summary(summary_train, step)
    testWrite.add_summary(summary_test, step)
    trainWrite.flush()
    testWrite.flush()


def train_fcn(trainPath, testPath, validPath):
    g = fcn_graph()
    images = g['images']
    labels = g['labels']
    learning_rate = g['learning_rate']
    train = g['train']
    loss = g['loss']
    logits = g['logits']
    with tf.Session() as sess:
        pwd = 'C:\\Users\\shiyx\\Documents\\CNNlog\FCN\\NO_bilinear&No_BN'
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        # saver.restore(sess,'/home/shi/CNNlog/FCN/FD/fullLayer07/para/')

        os.system('mkdir '+pwd)
        # command = ['mkdir ' + pwd,
        #            'rm -rf ' + pwd + '/*',
        #            'mkdir ' + pwd + '/code',
        #            'mkdir ' + pwd + '/sample',
        #            'cp -r /gpfs/share/home/1400012437/FaultDetection/fault-test/* ' + pwd + '/code']
        #
        # for i in command:
        #     os.system(i)

        trainWrite = tf.summary.FileWriter(pwd + '\\train', tf.get_default_graph())
        testWrite = tf.summary.FileWriter(pwd + '\\test', tf.get_default_graph())

        data = niiS(trainPath, testPath, expand=0)
        for i in range(5001):
            trainBatch = data.trainBatchLayer(1, shape=[250,400])
            # train
            feedData = {images: trainBatch[0].reshape([1,250,400,1]),
                        labels: trainBatch[1].reshape([1,250,400,1]),
                        learning_rate: 1e-3}

            fetchVariables = [loss, logits,train]

            [loss_i, logits_i, _] = sess.run(fetches=fetchVariables, feed_dict=feedData)

            if i % 100 == 0:
                save_summary_fcn(i, data, sess, trainWrite, testWrite, g)
            if i % 1000 == 0:
                a = logits_i[0, :, :, 0]*1
                pyplot.imshow(a.T)
                pyplot.savefig(pwd+'\\%s.png'%i)

            # if i % 15000 == 0 and i != 0:
            #     saver.save(sess, pwd + '\\para_%g\\' % i)
            #     print(1)
            print('Step:%s  loss:%s'%(i, loss_i))
        saver.save(sess, pwd + '\\para\\')
        trainWrite.close()
        testWrite.close()

    return sess



def save_summary_cnn(step, data, sess, trainWrite, testWrite, g):
    images = g['images']
    labels = g['labels']
    learning_rate = g['learning_rate']
    summary = g['summary']

    trainBatch = data.trainBatch(10, shape=[32, 32], addProb=0.2)
    # train
    feedData = {images: trainBatch[0].reshape([10, 32, 32, 1]),
                labels: trainBatch[1].reshape([10, 32, 32, 1]),
                learning_rate: 1e-3}
    fetchVariables = [summary]
    [summary_train] = sess.run(fetches=fetchVariables, feed_dict=feedData)
    testBatch = data.testBatch(10, shape=[32, 32], addProb=0.2)
    testBatch = [testBatch[0].reshape([10, 32, 32, 1]), testBatch[1].reshape([10, 32, 32, 1])]
    testFeed = {images: testBatch[0],
                labels: testBatch[1],
                learning_rate: 1e-3}
    [summary_test] = sess.run(fetches=fetchVariables, feed_dict=testFeed)

    trainWrite.add_summary(summary_train, step)
    testWrite.add_summary(summary_test, step)
    trainWrite.flush()
    testWrite.flush()


def train_cnn(trainPath, testPath, validPath):
    g = cnn_graph()
    images = g['images']
    labels = g['labels']
    learning_rate = g['learning_rate']
    train = g['train']
    loss = g['loss']
    logits = g['logits']
    with tf.Session() as sess:
        pwd = 'C:\\Users\\shiyx\\Documents\\CNNlog\\CNN\\test\\BN&pos=35'
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        # saver.restore(sess,'/home/shi/CNNlog/FCN/FD/fullLayer07/para/')

        os.system('mkdir '+pwd)
        # command = ['mkdir ' + pwd,
        #            'rm -rf ' + pwd + '/*',
        #            'mkdir ' + pwd + '/code',
        #            'mkdir ' + pwd + '/sample',
        #            'cp -r /gpfs/share/home/1400012437/FaultDetection/fault-test/* ' + pwd + '/code']
        #
        # for i in command:
        #     os.system(i)

        trainWrite = tf.summary.FileWriter(pwd + '\\train', tf.get_default_graph())
        testWrite = tf.summary.FileWriter(pwd + '\\test', tf.get_default_graph())

        data = niiS(trainPath, testPath)
        for i in range(50000):
            trainBatch = data.trainBatch(10, shape=[32, 32], addProb=0.2)
            # train
            feedData = {images: trainBatch[0].reshape([10, 32, 32, 1]),
                        labels: trainBatch[1].reshape([10, 32, 32, 1]),
                        learning_rate: 1e-3}

            fetchVariables = [loss, logits,train]

            [loss_i, logits_i, _] = sess.run(fetches=fetchVariables, feed_dict=feedData)

            if i % 100 == 0:
                save_summary_cnn(i, data, sess, trainWrite, testWrite, g)
            if i % 1000 == 0:
                a = logits_i[0, :].reshape([32, 32])
                b = trainBatch[0][0,:,:]+trainBatch[1][0,:,:]*0.5
                pyplot.subplot(121)
                pyplot.imshow(b.T)
                pyplot.subplot(122)
                pyplot.imshow(a.T)
                pyplot.savefig(pwd+'\\%s.png'%i)

            # if i % 15000 == 0 and i != 0:
            #     saver.save(sess, pwd + '\\para_%g\\' % i)
            #     print(1)
            print('Step:%s  loss:%s'%(i, loss_i))
        saver.save(sess, pwd + '\\para\\')
        trainWrite.close()
        testWrite.close()

    return sess

# train_fcn(validPath, validPath, validPath)
train_cnn(validPath, validPath, validPath)

