# -*- coding: utf-8 -*-

from __future__ import print_function

# Graph is built in Graph.py
import sys
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from run.get_train_data import *
sys.path.append('/home/pkushi/tremor')

MODEL = '1d1s3c'
data_train = Data_1d1s3c('/home/pkushi/dataset_1d1s3c')
data_test = Data_1d1s3c('/home/pkushi/dataset_1d1s3c_test')

from run.graph import *

NUM = 20

def save_summary_cnn(step, sess, trainWrite, testWrite, g):
    input = g['input']
    labels = g['labels']
    learning_rate = g['learning_rate']
    summary = g['summary']
    is_training = g['is_training']
    correct = g['correct']
    logits = g['logits']
    # train
    trainBatch = data_train.next_batch(NUM)
    feedData = {input: trainBatch[0],
                labels: trainBatch[1],
                learning_rate: 1e-3,
                is_training: True}
    fetchVariables = [summary, correct]
    [summary_train, correct_train] = sess.run(fetches=fetchVariables, feed_dict=feedData)
    # test
    mean = tf.get_collection('mean')[-1]
    moving_mean = tf.get_collection('moving_mean')[-1]
    var = tf.get_collection('var')[-1]
    fetchVariables = [summary, correct,mean, var, tf.arg_max(logits,1)]
    testBatch = data_test.next_batch(NUM)
    testFeed = {input: testBatch[0],
                labels: testBatch[1],
                learning_rate: 1e-3,
                is_training: False}
    [summary_test, correct_test, mean_i, var_i,result]= sess.run(fetches=fetchVariables, feed_dict=testFeed)

    print('***************************************************')
    print(correct_test)
    print(np.argmax(testBatch[1],1))
    print(result)
    print('***************************************************')
    print(mean_i)
    print(var_i)
    print('***************************************************')

    trainWrite.add_summary(summary_train, step)
    testWrite.add_summary(summary_test, step)
    trainWrite.flush()
    testWrite.flush()


def plot_2d4s1c(data, logits, step, pwd):
    wave = data[0]
    label = data[1]
    result = logits
    for i in range(NUM):
        plt.figure()
        plt.plot(np.linspace(0,400,40000), wave[i,0,:])
        plt.plot(np.linspace(0,400,40000), wave[i,1,:]+4)
        plt.plot(np.linspace(0,400,40000), wave[i,2,:]+8)
        plt.plot(np.linspace(0,400,40000), wave[i,3,:]+12)

        label_i = label[i,:].argmax()
        if label_i==0:
            label_i='tremor'
        elif label_i==2:
            label_i='noise'
        else:
            label_i='EQ'
        result_i = result[i, :].argmax()
        if result_i == 0:
            result_i = 'tremor'
        elif result_i == 2:
            result_i = 'noise'
        else:
            result_i = 'EQ'

        plt.title('label:'+label_i+',result:'+result_i)
        plt.savefig(pwd+'/%s_%s.png'%(step, i))


def plot_1d1s3c(data, logits, step, pwd):
    wave = data[0]
    label = data[1]
    result = logits
    for i in range(NUM):
        plt.figure()
        plt.plot(np.linspace(0,200,20000), wave[i,:, 0]+0, label='E')
        plt.plot(np.linspace(0,200,20000), wave[i,:, 1]+4, label='N')
        plt.plot(np.linspace(0,200,20000), wave[i,:, 2]+8, label='U')

        label_i = label[i,:].argmax()
        if label_i==0:
            label_i='tremor'
        elif label_i==2:
            label_i='noise'
        else:
            label_i='EQ'
        result_i = result[i, :].argmax()
        if result_i == 0:
            result_i = 'tremor'
        elif result_i == 2:
            result_i = 'noise'
        else:
            result_i = 'EQ'

        plt.title('label:'+label_i+',result:'+result_i)
        plt.savefig(pwd+'/%s_%s.png'%(step, i))


def plot_1d3s3c(data, logits, step, pwd):
    wave = data[0].T
    label = data[1]
    result = logits
    for i in range(NUM):
        plt.figure()
        plt.plot(np.linspace(0,200,20000), wave[0, :], label='1sE')
        plt.plot(np.linspace(0,200,20000), wave[1, :] + 3, label='1sN')
        plt.plot(np.linspace(0,200,20000), wave[2, :] + 6, label='1sU')
        plt.plot(np.linspace(0,200,20000), wave[3, :] + 9, label='2sE')
        plt.plot(np.linspace(0,200,20000), wave[4, :] + 12, label='2sN')
        plt.plot(np.linspace(0,200,20000), wave[5, :] + 15, label='2sU')
        plt.plot(np.linspace(0,200,20000), wave[6, :] + 18, label='3sE')
        plt.plot(np.linspace(0,200,20000), wave[7, :] + 21, label='3sN')
        plt.plot(np.linspace(0,200,20000), wave[8, :] + 24, label='3sU')

        label_i = label[i,:].argmax()
        if label_i==0:
            label_i='tremor'
        elif label_i==2:
            label_i='noise'
        else:
            label_i='EQ'
        result_i = result[i, :].argmax()
        if result_i == 0:
            result_i = 'tremor'
        elif result_i == 2:
            result_i = 'noise'
        else:
            result_i = 'EQ'

        plt.title('label:'+label_i+',result:'+result_i)
        plt.savefig(pwd+'/%s_%s.png'%(step, i))

def train_cnn():
    g = cnn_graph_1d()
    '''
    g = {
        'input': input,
        'labels': labels_r,
        'learning_rate': learning_rate,
        'logits': logits,
        'loss': loss,
        'train': train,
        'summary': summary,
        'is_training': is_training
    }
    '''
    input = g['input']
    labels = g['labels']
    learning_rate = g['learning_rate']
    train = g['train']
    loss = g['loss']
    logits = g['logits']
    is_training = g['is_training']
    ops = tf.get_collection('ops')
    with tf.Session() as sess:
        pwd = '/home/pkushi/CNNlog/CNN/TNE_1d1s3c_test1'
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        #saver.restore(sess,'/home/pkushi/CNNlog/CNN/tremor_with_norm_BN_test5/para_6000/')

        command = ['mkdir ' + pwd]
        for i in command:
            os.system(i)

        trainWrite = tf.summary.FileWriter(pwd + '/train', tf.get_default_graph())
        testWrite = tf.summary.FileWriter(pwd + '/test', tf.get_default_graph())
        plot_f = {'1d1s3c':plot_1d1s3c, '1d3s3c':plot_1d3s3c, '2d4s1c':plot_2d4s1c}
        plot = plot_f[MODEL]
        for i in range(1000000):
            trainBatch = data_train.next_batch(NUM)
            # train
            feedData = {input: trainBatch[0],
                        labels: trainBatch[1],
                        learning_rate: 1e-3,
                        is_training: True}
            fetchVariables = [loss, logits, train,ops]
            if i % 10 == 0:
                save_summary_cnn(i, sess, trainWrite, testWrite, g)

            [loss_i, logits_i, _, _] = sess.run(fetches=fetchVariables, feed_dict=feedData)

            if i % 3000 == 0 and i != 0:
                saver.save(sess, pwd + '/para_%g/' % i)
                plot(trainBatch, logits_i, i, pwd)
                print(1)
            print('Step:%s  loss:%s'%(i, loss_i))
            #for j in range(NUM):
            #    print('label:%s, logits:%s'%(trainBatch[1][j,:].argmax(), logits_i[j,:].argmax()))
            #plot(trainBatch,logits_i,i,pwd)
        saver.save(sess, pwd + '/para/')
        trainWrite.close()
        testWrite.close()

    return sess


# train_fcn(validPath, validPath, validPath)
train_cnn()


