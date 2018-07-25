from unit.conv import *
from unit.NN import *
from unit.pool import *
# from fcn_config import *
from run.cnn_config import *


def cnn_graph_2d():
    # 读取指定的CNN网络的参数
    conf = CNNConf_2d4s1c()
    INPUT_SHAPE = conf.INPUT_SHAPE
    OUTPUT_SHAPE = conf.OUTPUT_SHAPE

    with tf.name_scope('ImagesLabels'):
        images = tf.placeholder(tf.float32, shape=[None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
        labels_r = tf.placeholder(tf.float32, shape=[None, OUTPUT_SHAPE[0]])
        labels = labels_r
        learning_rate = tf.placeholder(tf.float32, name='LR')
        is_training = tf.placeholder(tf.bool, name='train_control')

    conf.set_is_training(is_training)
    # convolution block and pool
    feature1 = conv2d_block(images, conf.conv[0])
    pool1 = max_pool(feature1, name='pool1', conf=conf.pool[0])

    feature2 = conv2d_block(pool1, conf.conv[1])
    pool2 = max_pool(feature2, name='pool2', conf=conf.pool[1])

    feature3 = conv2d_block(pool2, conf.conv[2])
    pool3 = max_pool(feature3, name='pool3', conf=conf.pool[2])

    feature4 = conv2d_block(pool3, conf.conv[3])
    pool4 = max_pool(feature4, name='pool4', conf=conf.pool[3])
    # flat and NN   layers
    with tf.name_scope('Flat'):
        w = pool4.shape.as_list()[1]
        h = pool4.shape.as_list()[2]
        c = pool4.shape.as_list()[3]
        x = tf.reshape(pool4, [-1, w*h*c])

    nn1 = nn_layer(x, conf.NN[0])
    nn2 = nn_layer(nn1, conf.NN[1])
    model_output = nn_layer(nn2, conf.NN[2])

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=model_output))

    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('Evaluate'):
        logits = tf.nn.softmax(model_output)
        model_result = tf.argmax(logits, 1)
        label_result = tf.argmax(labels, 1)
        correct = tf.reduce_mean(tf.cast(tf.equal(model_result, label_result), tf.float32))



    with tf.name_scope('Summary'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('correct', correct)
        tf.summary.histogram('conv2',feature2)
        tf.summary.histogram('conv4',feature4)

        tf.summary.histogram('nn1', nn1)
        tf.summary.histogram('nn2', nn2)
        tf.summary.histogram('output', model_output)

        summary = tf.summary.merge_all()
    ops = tf.get_collection('ops')
    value = {
        'input': images,
        'labels': labels_r,
        'learning_rate': learning_rate,
        'logits': logits,
        'loss': loss,
        'train': train,
        'summary': summary,
        'is_training': is_training,
        'ops': ops
    }

    return value


def cnn_graph_1d():
    # 读取指定的CNN网络的参数
    conf = CNNConf_1d1s3c()
    INPUT_SHAPE = conf.INPUT_SHAPE
    OUTPUT_SHAPE = conf.OUTPUT_SHAPE

    with tf.name_scope('ImagesLabels'):
        wave = tf.placeholder(tf.float32, shape=[None, INPUT_SHAPE[0], INPUT_SHAPE[1]])
        labels_r = tf.placeholder(tf.float32, shape=[None, OUTPUT_SHAPE[0]])
        labels = labels_r
        learning_rate = tf.placeholder(tf.float32, name='LR')
        is_training = tf.placeholder(tf.bool, name='train_control')

    conf.set_is_training(is_training)

    # convolution block and pool
    assert len(conf.conv) == len(conf.pool)
    conv_feature = []
    feature = wave
    for i in range(len(conf.conv)):
        feature = conv1d_block(feature, conf.conv[i])
        conv_feature.append(feature)
        tf.add_to_collection('feature', feature)
        feature = max_pool_1d(feature, name='pool%s'%i, conf=conf.pool[i])

    # flat and NN layers
    with tf.name_scope('Flat'):
        w = feature.shape.as_list()[1]
        c = feature.shape.as_list()[2]
        x = tf.reshape(feature, [-1, w*c])


    nn1 = nn_layer(x, conf.NN[0])
    nn2 = nn_layer(nn1, conf.NN[1])
    model_output = nn_layer(nn2, conf.NN[2])

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=model_output))

    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('Evaluate'):
        logits = tf.nn.softmax(model_output)
        model_result = tf.argmax(logits, 1)
        label_result = tf.argmax(labels, 1)
        correct = tf.reduce_mean(tf.cast(tf.equal(model_result, label_result), tf.float32))



    with tf.name_scope('Summary'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('correct', correct)
        tf.summary.histogram('convb3', conv_feature[3])
        tf.summary.histogram('nn2', nn2)
        tf.summary.histogram('output', model_output)

        summary = tf.summary.merge_all()
    ops = tf.get_collection('ops')
    value = {
        'input': wave,
        'labels': labels_r,
        'learning_rate': learning_rate,
        'logits': logits,
        'loss': loss,
        'train': train,
        'summary': summary,
        'is_training': is_training,
        'ops': ops
    }

    return value
