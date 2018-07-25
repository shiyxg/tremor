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

from dataAnalysis.nii import *
from FCN.graph import *
from FCN.path import *
## libs
# change the learning rate with accuracy & loss & i
def learn_rate(accuracy, loss, i):
    LR_i = 1e-4
    '''
    loss = np.log(loss)
    if 0<= loss:
        LR_i = 1e-4
    elif -2<=loss < 0:
        LR_i = 5e-4        
    elif -3 <= loss<-2:   
        LR_i = 2e-4
    elif loss<-3:
        LR_i = 1e-4
    '''
    return LR_i

def visual(data,sess,name,shape,graph):
    resultLabel = graph.get_collection('resultLabel')[0]
    images_ori  = graph.get_collection('images')[0]
    labels_ori  = graph.get_collection('labels')[0]
    keepProb    = graph.get_collection('keepProb')[0]
    LR          = graph.get_collection('learningRate')[0]
    [XL,YL,TL]  = data.shape
    expand = data.expand
    layerIndex = [YL//2]
    size = np.array([len(layerIndex),XL+expand*2,TL+expand*2])
    sampleSize = [XL,TL]
    labelSample = np.zeros(size)
    labelOri = np.zeros(size)
    dataOri  = np.zeros(size)
    for i in range(len(layerIndex)):
        
        
        [a,b] = data.pickLayer(layerIndex[i],sampleAxis=13)
        dataOri[i, expand:(size[1]-expand),expand:(size[2]-expand)]  = a
        labelOri[i,expand:(size[1]-expand),expand:(size[2]-expand)] = b
        feedData ={images_ori:np.reshape(a,[1,shape[0],shape[1]]),            
                    labels_ori:np.reshape(b,[1,shape[0],shape[1]]),
                    keepProb:1,LR:1e-5}
        result_ijk = sess.run(resultLabel,feed_dict = feedData)
        labels_ijk = result_ijk.reshape(shape)
        labelSample[i,:,:] = labels_ijk
       
    dataOri = dataOri[  :, expand:(size[1]-expand),expand:(size[2]-expand)]
    labelOri = labelOri[:, expand:(size[1]-expand),expand:(size[2]-expand)]
    labelSample = labelSample[:,expand:(size[1]-expand),expand:(size[2]-expand)]
    for i in range(len(layerIndex)):       
        train = np.reshape(dataOri[i,:,:] + 3*labelSample[i,:,:], sampleSize)
        ori = np.reshape(dataOri[i,:,:] + 3*labelOri[i,:,:],    sampleSize)
        label = np.reshape(labelSample[i,:,:],sampleSize)
        pyplot.pcolor(train.T)
        pyplot.savefig(name+'all.png')
        pyplot.pcolor(label.T)
        pyplot.savefig(name+'label.png')
        
def result_visual(dataSets,sess,pwd,step,shape,graph = tf.get_default_graph()):
    
    for data in dataSets.train:
        name = pwd+'/train/'+data.model+'_%g_'%step
        visual(data,sess,name,shape,graph)
    
    for data in dataSets.test:
        name = pwd+'/test/'+data.model+'_%g_'%step
        visual(data,sess,name,shape,graph)
        
        
    
def saveSamples(i,batch,pwd,shape):
    [record,label] = batch
    assert len(record)==len(label)
    
    for j in range(len(record)):
        a = np.reshape(record[j,:,:],shape)
        b = np.reshape(label[j,:,:], shape)
        #figure = a.T + b.T*10
        pyplot.imshow(a.T,cmap=cm.gray)
        pyplot.savefig(pwd+'/sample/%g%g_train_record.png'%(i,j))
        
        pyplot.imshow(a.T+b.T*128*3,cmap=cm.gray)
        pyplot.savefig(pwd+'/sample/%g%g_train_all.png'%(i,j))

def saveSummary(step,data,sess,pwd,add,shape,trainWrite,testWrite,g=tf.get_default_graph()):
    
    images = g.get_collection('images')[0]
    labels = g.get_collection('labels')[0]
    LR     = g.get_collection('learningRate')[0]
    keepProb = g.get_collection('keepProb')[0]
    
    [correct,error_f,error_nf]   = g.get_collection('precision')[0]
    summary = g.get_collection('summary')[0]
    loss = g.get_collection('loss')[0]
    
    
    trainBatch = data.trainBatchLayer(1,shape=shape)
    testBatch  = data.testBatchLayer(1 ,shape=shape)
    
    trainFeed = {images:trainBatch[0], labels:trainBatch[1],LR:1e-4, keepProb:1}    
    testFeed  = {images:testBatch[0],  labels:testBatch[1], LR:1e-4, keepProb:1}
    
    fetchVariables = [correct,   error_f, error_nf,   loss,     summary]
    
    [correct_train,  error_f_train,    error_nf_train, loss_train, summary_train
    ] = sess.run(fetches = fetchVariables, feed_dict = trainFeed)
    [correct_test,   error_f_test,     error_nf_test,  loss_test,  summary_test
    ] = sess.run(fetches = fetchVariables, feed_dict = testFeed)
        
    trainWrite.add_summary(summary_train,step)
    testWrite.add_summary(summary_test,step)
    trainWrite.flush()
    testWrite.flush()
    
    print('Loss in Train: %g'%loss_train)
    print('loss in Test: %g'%loss_test)
    accu_test = [correct_test,error_f_test,error_nf_test]
    LR_i = learn_rate(accu_test,loss_test,step)
    
    return LR_i
    
## train & test
def train(trainPath,testPath,validPath):
    ## build a graph
    LR_i=1e-4
    batch_i = 10
    add = 0.8
    keep = 0.5
    shape=[250,400]
    
    [g,para] = Graph(inputShape=shape)

    '''
    para = {
            'images':images_ori, 'labels':labels_ori,
            'learningRate':LR,         'keepProb':keepProb,
            
            'result':result,     
            'trainStep':trainStep,
            
            'precision':[correct, error_f, error_nf], 
            'loss':loss
            'summary':summary,
            'faultNum':faultNum
            }
    '''
    
    images = g.get_collection('images')[0]
    labels = g.get_collection('labels')[0]
    LR     = g.get_collection('learningRate')[0]
    keepProb = g.get_collection('keepProb')[0]
    
    trainStep = g.get_collection('trainStep')[0]
    [correct,error_f,error_nf]   = g.get_collection('precision')[0]
    summary = g.get_collection('summary')[0]
    loss = g.get_collection('loss')[0]
    faultNum = g.get_collection('faultNum')[0]
    resultLabel = g.get_collection('resultLabel')[0]
    
    ## data
    #path = 'C:\\Users\\Shi\\Documents\\FaultDetection\\data\\Synthetic\\FCN1_OA3_f10_n10\\Nifti'
    
    data = niiS(trainPath,testPath,expand = 0)
    #ata = niiS(['/home/shi/FaultDetection/data/SYN/FCN1_OA3_f10_n10/Nifti'],['/home/shi/FaultDetection/data/SYN/FCN2_OA3_f40_n10/Nifti'],expand = 0)
    ## train
    with tf.Session(graph=g) as sess:
        pwd = '/gpfs/share/home/1400012437/CNNlog/FCN/PKU/04'
        #pwd = 'C:\\Users\\Shi\\Documents\\log\\test'
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()
        #saver.restore(sess,'/home/shi/CNNlog/FCN/FD/fullLayer07/para/')

        command=['mkdir '+pwd,
                 'rm -rf '+pwd+'/*',
                 'mkdir '+pwd+'/code',
                 'mkdir '+pwd+'/sample',
                 'cp -r /gpfs/share/home/1400012437/FaultDetection/fault-test/* '+pwd+'/code']

                 
        for i in command:
            os.system(i)
        
        trainWrite = tf.summary.FileWriter(pwd+'/train', tf.get_default_graph())
        testWrite  = tf.summary.FileWriter(pwd+'/test', tf.get_default_graph())
        
    
        for i in range(50001):
            trainBatch = data.trainBatchLayer(batch_i,shape=shape)
            
            # output the samples
            if i==0:
                #saveSamples(i,trainBatch,pwd,shape)
                print(1)
            #train
            feedData = {images:trainBatch[0], labels:trainBatch[1],
                        keepProb:keep,       LR:LR_i}
                        
            fetchVariables = [trainStep]
            #result_visual(data,sess,pwd,i,shape,graph = g)
            
            [_] = sess.run(fetches = fetchVariables, feed_dict = feedData)
            
            if i%100==0:
                LR_i = saveSummary(i,data,sess,pwd,add,shape,trainWrite, testWrite, g=g)
                print(i)
            if i%15000 == 0 and i !=0:
                result_visual(data,sess,pwd,i,shape,graph = g)
                print(1)
            if i%15000 == 0 and i!=0:
                saver.save(sess,pwd+'/para_%g/'%i)
        saver.save(sess,pwd+'/para/')
        trainWrite.close()
        testWrite.close()
        
    return sess;
        
        
    
    
    
    
    ## train
    
a = train(trainPath,testPath,validPath)
    
    
