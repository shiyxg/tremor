# -*- coding: utf-8 -*-

from __future__ import print_function

# Graph is built in Graph.py
import sys
import time
sys.path.append('/home/pkushi/tremor')
from run.get_train_data import *
from run.plot import *
from run.graph import *


MODEL = 'T_1d1s3c'
GRAPH = {'1d1s3c':cnn_graph_1d, 'T_1d1s3c':cnn_graph_1d, '2d4s1c':cnn_graph_2d, '1d3s3c':cnn_graph_1d}
DATASET = {'1d1s3c':Data_1d1s3c, '2d4s1c':Data_2d4s1c, 'T_1d1s3c':Data_T_1d1s3c,'1d3s3c':Data_1d3s3c}
Data = DATASET[MODEL]
data_train = Data('/home/pkushi/dataset_'+MODEL)
data_test = Data('/home/pkushi/dataset_'+MODEL+'_test')
# data_test = Data_simple('/home/pkushi/datasetdecay0.999_T_1d1s3c/aichi')
NUM = 1
SAMPLE_NUM = 10000
IS_TRAINING = True
LOG_PATH = '/home/pkushi/CNNlog/success/TTT_1d1s3c_chn5_100k/para/'
pwd = '/home/pkushi/CNNlog/CNN/feature/TTT/test_error1'
# LOG_PATH = '/home/pkushi/CNNlog/success/TNE_1d1s3c_800k/para_100000/'
# pwd = '/home/pkushi/CNNlog/CNN/feature/TNE_1d1s3c/test_error3'
# LOG_PATH = '/home/pkushi/CNNlog/CNN/tremor_with_event_nostor/para_12000/'
# pwd = '/home/pkushi/CNNlog/CNN/BN_test_event_2d4s1c/noise_filter'
# LOG_PATH = '/home/pkushi/CNNlog/CNN/tremor_with_event_nostor/para_12000/'
# pwd = '/home/pkushi/CNNlog/CNN/BN_test_event_2d4s1c/noise_filter'

def save_summary_cnn(step, sess, trainWrite, testWrite, g):
    input = g['input']
    labels = g['labels']
    learning_rate = g['learning_rate']
    summary = g['summary']

    # train
    trainBatch = data_train.next_batch(NUM)
    feedData = {input: trainBatch[0],
                labels: trainBatch[1],
                learning_rate: 1e-3}
    fetchVariables = [summary]
    [summary_train] = sess.run(fetches=fetchVariables, feed_dict=feedData)
    # test
    testBatch = data_test.next_batch(NUM)
    testFeed = {input: testBatch[0],
                labels: testBatch[1],
                learning_rate: 1e-3}
    [summary_test] = sess.run(fetches=fetchVariables, feed_dict=testFeed)

    trainWrite.add_summary(summary_train, step)
    testWrite.add_summary(summary_test, step)
    trainWrite.flush()
    testWrite.flush()


def train_cnn():
    graph = GRAPH[MODEL]
    g = graph()
    '''
    g = {
        'input': input,
        'labels': labels_r,
        'learning_rate': learning_rate,
        'logits': logits,
        'loss': loss,
        'train': train,
        'summary': summary
    }
    '''
    input = g['input']
    labels = g['labels']
    learning_rate = g['learning_rate']
    train = g['train']
    loss = g['loss']
    logits = g['logits']
    is_training = g['is_training']
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess,LOG_PATH)

        command = ['mkdir ' + pwd]
        for i in command:
            os.system(i)

        # trainWrite = tf.summary.FileWriter(pwd + '/train', tf.get_default_graph())
        # testWrite = tf.summary.FileWriter(pwd + '/test', tf.get_default_graph())

        mean = tf.get_collection('mean')[-1]
        var = tf.get_collection('var')[-1]
        BN_reuslt = tf.get_collection('BN_result')[-1]
        wx_result = tf.get_collection('wx_result')[-1]
        conv_result = tf.get_collection('conv_result')
        feature = tf.get_collection('feature')
        fetchVariables = [loss, logits, mean, var, BN_reuslt, wx_result, conv_result, feature]

        eva = np.zeros([3,3])
        plot_f = {'1d1s3c':plot_1d1s3c, '2d4s1c':plot_2d4s1c, '1d3s3c':plot_1d3s3c,'T_1d1s3c':plot_T_1d1s3c}
        plot = plot_f[MODEL]

        for i in range(SAMPLE_NUM):
            start = time.time()
            trainBatch = data_test.next_batch(NUM)
            end1 = time.time()
            # train
            feedData = {input: trainBatch[0],
                        labels: trainBatch[1],
                        learning_rate: 1e-3,
                        is_training: False}

            [loss_i, logits_i, mean_i, var_i, BN_i, wx_i, conv_result_i, feature_i] = sess.run(fetches=fetchVariables, feed_dict=feedData)
            result_i = logits_i.argmax(1)
            label_i = trainBatch[1].argmax(1)
            end2 = time.time()

            eva[int(result_i)][int(label_i)] = eva[int(result_i)][int(label_i)] + 1

            if i % 10 == 0:
                # save_summary_cnn(i, sess, trainWrite, testWrite, g)
                print(1)
            if i % 15000 == 0 and i != 0:
                saver.save(sess, pwd + '/para_%g/' % i)
                print(1)

            print('************************************************')
            print('Step:%s  loss:%s'%(i, loss_i))
            print(label_i)
            print(result_i)
            print(logits_i)
            print('data pick take: %s s'%(end1-start))
            print('model computing take: %s s' % (end2 - end1))

            if 0 or label_i != result_i:
                plot(trainBatch, logits_i, i, pwd)
                plot_features(feature_i, pwd+'/%s'%i)
            # features(trainBatch[0], conv_result_i)
            # print('_________________________________________________')
            # print(mean_i)
            # print(var_i)
            # print('__________________________________________________')
            # print(BN_i)
            # print(wx_i)
            print('##################################################')

            print(eva)
        # saver.save(sess, pwd + '/para/')
        # trainWrite.close()
        # testWrite.close()

    return sess


# train_fcn(validPath, validPath, validPath)
train_cnn()
