from unit.conv import *
from unit.deconv import *
from unit.NN import *
from unit.loss import *
from unit.pool import *
from fcn_config import *


# the following is some graph *(for example, fcn), the config file is *_config.py


def fcn_graph():
    '''
    :return: the graph like the FCN paper in
    Shelhamer E, Long J, Darrell T. Fully Convolutional Networks for Semantic Segmentation[J]. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2017, 39(4):640-651.
    '''
    # input
    conf = FCNConf1()
    INPUT_SHAPE = conf.INPUT_SHAPE
    OUTPUT_SHAPE = conf.OUTPUT_SHAPE
    with tf.name_scope('ImagesLabels'):
        images = tf.placeholder(tf.float32, shape=[None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
        labels = tf.placeholder(tf.float32, shape=[None, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1], OUTPUT_SHAPE[2]])
        learning_rate = tf.placeholder(tf.float32, name='LR')

    # convolution block and pool
    feature1 = conv2d_block(images,   conf.conv1)
    pool1 = max_pool(feature1, name='pool1')

    feature2 = conv2d_block(pool1, conf.conv2)
    pool2 = max_pool(feature2, name='pool2')

    feature3 = conv2d_block(pool2, conf.conv3)
    pool3 = max_pool(feature3, name='pool3')

    feature4 = conv2d_block(pool3, conf.conv4)

    # deconvolution
    deconv1 = deconv2d_layer(feature1, conf.deconv1)
    deconv2 = deconv2d_layer(feature2, conf.deconv2)
    deconv3 = deconv2d_layer(feature3, conf.deconv3)
    deconv4 = deconv2d_layer(feature4, conf.deconv4)
    # connect all deconv result
    with tf.name_scope('Fuse'):
        deconv_fuse = tf.concat([deconv1, deconv2, deconv3, deconv4], 3, name='FuseTo1Imag')
        deconv_fuse = tf.reshape(deconv_fuse, [-1, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1], 4])
        model_output = conv2d_layer(deconv_fuse, conf.fuse)
        logits = tf.sigmoid(model_output)
    # evalute the distance between labels and output
    loss = weight_cross_entropy(labels, model_output)

    # train and change the parameters
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('Evaluate'):
        result = tf.round(logits)
        sub = labels - result
        correct_all = tf.reduce_mean(tf.cast(tf.equal(sub, 0.0), tf.float32))

        pos = tf.reduce_sum(tf.cast(tf.equal(labels, 1.0), tf.float32))
        neg = tf.reduce_sum(tf.cast(tf.equal(labels, 0.0), tf.float32))
        correct_pos = 1 - tf.reduce_sum(tf.cast(tf.equal(sub, 1.0), tf.float32)) / pos
        correct_neg = 1 - tf.reduce_sum(tf.cast(tf.equal(sub, -1.0), tf.float32)) / neg

    with tf.name_scope('Summary'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('correct_all', correct_all)
        tf.summary.scalar('correct_pos', correct_pos)
        tf.summary.scalar('correct_neg', correct_neg)
        tf.summary.image('input', images)
        tf.summary.image('logits', logits)
        tf.summary.image('deconv1', deconv1)
        tf.summary.image('deconv2', deconv2)
        tf.summary.image('deconv3', deconv3)
        tf.summary.image('deconv4', deconv4)

        summary = tf.summary.merge_all()

    Value = {
        'images': images,
        'labels': labels,
        'learning_rate': learning_rate,
        'logits': logits,
        'loss': loss,
        'train': train_step,
        'summary': summary
    }

    return Value


def cnn_graph():
    # 读取指定的CNN网络的参数
    conf = CNNConf()
    INPUT_SHAPE = conf.INPUT_SHAPE
    OUTPUT_SHAPE = conf.OUTPUT_SHAPE

    with tf.name_scope('ImagesLabels'):
        images = tf.placeholder(tf.float32, shape=[None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
        labels_r = tf.placeholder(tf.float32, shape=[None, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1], OUTPUT_SHAPE[2]])
        labels = tf.reshape(labels_r, [-1, OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1] * OUTPUT_SHAPE[2]])
        learning_rate = tf.placeholder(tf.float32, name='LR')

    # convolution block and pool
    feature1 = conv2d_block(images, conf.conv[0])
    pool1 = max_pool(feature1, name='pool1')

    feature2 = conv2d_block(pool1, conf.conv[1])
    pool2 = max_pool(feature2, name='pool2')

    feature3 = conv2d_block(pool2, conf.conv[2])
    pool3 = max_pool(feature3, name='pool3')

    feature4 = conv2d_block(pool3, conf.conv[3])

    # flat and NN   layers
    with tf.name_scope('Flat'):
        w = feature4.shape.as_list()[1]
        h = feature4.shape.as_list()[2]
        c = feature4.shape.as_list()[3]
        x = tf.reshape(feature4, [-1, w*h*c])

    nn1 = nn_layer(x, conf.NN[0])
    nn2 = nn_layer(nn1, conf.NN[1])
    model_output = nn_layer(nn2, conf.NN[2])

    logits = tf.sigmoid(model_output)
    # loss = sigmoid_cross_entropy(label=labels, logits=model_output)
    loss = weight_cross_entropy(labels, model_output, pos=35)
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('Evaluate'):
        result = tf.round(logits)
        sub = labels - result
        correct_all = tf.reduce_mean(tf.cast(tf.equal(sub, 0.0), tf.float32))

        # pos = tf.reduce_sum(tf.cast(tf.equal(labels, 1.0), tf.float32))
        # neg = tf.reduce_sum(tf.cast(tf.equal(labels, 0.0), tf.float32))
        #
        # # control if pos==0 or neg==0
        # tf.cond(pos == 0, lambda:1, lambda:pos)
        # tf.cond(neg == 0, lambda:1, lambda:neg)
        correct_pos = tf.reduce_sum(tf.cast(tf.equal(sub, 1.0), tf.float32))
        correct_neg = tf.reduce_sum(tf.cast(tf.equal(sub, -1.0), tf.float32))

    with tf.name_scope('Summary'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('correct_all', correct_all)
        tf.summary.scalar('correct_pos', correct_pos)
        tf.summary.scalar('correct_neg', correct_neg)
        tf.summary.histogram('conv2',feature2)
        tf.summary.histogram('conv4',feature4)

        tf.summary.histogram('nn1', nn1)
        tf.summary.histogram('nn2', nn2)
        tf.summary.histogram('nn3', model_output)
        tf.summary.image('input', images)
        tf.summary.image('output', tf.reshape(result, [-1, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1], OUTPUT_SHAPE[2]]))
    summary = tf.summary.merge_all()

    Value = {
        'images': images,
        'labels': labels_r,
        'learning_rate': learning_rate,
        'logits': logits,
        'loss': loss,
        'train': train,
        'summary': summary
    }

    return Value
