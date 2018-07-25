# -*- coding:utf-8


class CNNConf_2d4s1c:
    def __init__(self, is_training=None):
        self.INPUT_SHAPE = [4, 40000, 1]
        self.OUTPUT_SHAPE = [3]

        self.conv = []
        self.conv.append({
            'name': 'convBlock1',
            'times': 2,
            'filterSize': [4, 11],
            'outputChn': 64,
            'strideSize': [1, 1],
            'is_training': is_training
        })
        self.conv.append({
            'name': 'convBlock2',
            'times': 2,
            'filterSize': [4, 7],
            'outputChn': 128,
            'strideSize': [1, 1],
            'is_training': is_training
        })
        self.conv.append({
            'name': 'convBlock3',
            'times': 2,
            'filterSize': [4, 5],
            'outputChn': 256,
            'strideSize': [1, 1],
            'is_training': is_training
        })
        self.conv.append({
            'name': 'convBlock4',
            'times': 2,
            'filterSize': [4, 5],
            'outputChn': 512,
            'strideSize': [1, 1],
            'is_training': is_training
        })

        self.NN = []
        self.NN.append({
            'name': 'NN1',
            'num': 4096,
            'is_training': is_training
        })
        self.NN.append({
            'name': 'NN2',
            'num': 4096,
            'is_training': is_training
        })
        self.NN.append({
            'name': 'output',
            'num': self.OUTPUT_SHAPE[0],
            'is_training': is_training
        })

        self.pool = []
        self.pool.append({
            'ksize': [1, 8, 4, 1],
            'stride': [1, 1, 8, 1]
        })
        self.pool.append({
            'ksize': [1, 8, 4, 1],
            'stride': [1, 1, 8, 1]
        })
        self.pool.append({
            'ksize': [1, 8, 4, 1],
            'stride': [1, 1, 8, 1]
        })
        self.pool.append({
            'ksize': [1, 8, 4, 1],
            'stride': [1, 1, 8, 1]
        })

    def set_is_training(self, is_training):
        for i in self.conv:
            i['is_training'] = is_training
        for i in self.NN:
            i['is_training'] = is_training


class CNNConf_1d1s3c:
    def __init__(self, is_training=None):
        self.INPUT_SHAPE = [20000, 3]
        self.OUTPUT_SHAPE = [3]

        self.conv = []
        self.conv.append({
            'name': 'convBlock1',
            'times': 1,
            'filterSize': [13],
            'outputChn': 64,
            'strideSize': [1],
            'is_training': is_training
        })
        self.conv.append({
            'name': 'convBlock2',
            'times': 1,
            'filterSize': [9],
            'outputChn': 128,
            'strideSize': [1],
            'is_training': is_training
        })
        self.conv.append({
            'name': 'convBlock3',
            'times': 2,
            'filterSize': [5],
            'outputChn': 256,
            'strideSize': [1],
            'is_training': is_training
        })
        self.conv.append({
            'name': 'convBlock4',
            'times': 2,
            'filterSize': [5],
            'outputChn': 512,
            'strideSize': [1],
            'is_training': is_training
        })
        self.conv.append({
            'name': 'convBlock5',
            'times': 1,
            'filterSize': [3],
            'outputChn': 1024,
            'strideSize': [1],
            'is_training': is_training
        })

        self.NN = []
        self.NN.append({
            'name': 'NN1',
            'num': 2048,
            'is_training': is_training
        })
        self.NN.append({
            'name': 'NN2',
            'num': 2048,
            'is_training': is_training
        })
        self.NN.append({
            'name': 'output',
            'num': self.OUTPUT_SHAPE[0],
            'is_training': is_training
        })

        self.pool = []
        self.pool.append({
            'ksize': [1, 8, 1],
            'stride': [1, 8, 1]
        })
        self.pool.append({
            'ksize': [1, 8, 1],
            'stride': [1, 8, 1]
        })
        self.pool.append({
            'ksize': [1, 4, 1],
            'stride': [1, 4, 1]
        })
        self.pool.append({
            'ksize': [1, 4, 1],
            'stride': [1, 4, 1]
        })
        self.pool.append({
            'ksize': [1, 4, 1],
            'stride': [1, 4, 1]
        })

    def set_is_training(self, is_training):
        for i in self.conv:
            i['is_training'] = is_training
        for i in self.NN:
            i['is_training'] = is_training


class CNNConf_1d3s3c:
    def __init__(self, is_training=None):
        self.INPUT_SHAPE = [20000, 9]
        self.OUTPUT_SHAPE = [3]

        self.conv = []
        self.conv.append({
            'name': 'convBlock1',
            'times': 1,
            'filterSize': [13],
            'outputChn': 64,
            'strideSize': [1],
            'is_training': is_training
        })
        self.conv.append({
            'name': 'convBlock2',
            'times': 1,
            'filterSize': [9],
            'outputChn': 128,
            'strideSize': [1],
            'is_training': is_training
        })
        self.conv.append({
            'name': 'convBlock3',
            'times': 2,
            'filterSize': [5],
            'outputChn': 256,
            'strideSize': [1],
            'is_training': is_training
        })
        self.conv.append({
            'name': 'convBlock4',
            'times': 2,
            'filterSize': [5],
            'outputChn': 512,
            'strideSize': [1],
            'is_training': is_training
        })
        self.conv.append({
            'name': 'convBlock5',
            'times': 1,
            'filterSize': [3],
            'outputChn': 1024,
            'strideSize': [1],
            'is_training': is_training
        })

        self.NN = []
        self.NN.append({
            'name': 'NN1',
            'num': 2048,
            'is_training': is_training
        })
        self.NN.append({
            'name': 'NN2',
            'num': 2048,
            'is_training': is_training
        })
        self.NN.append({
            'name': 'output',
            'num': self.OUTPUT_SHAPE[0],
            'is_training': is_training
        })

        self.pool = []
        self.pool.append({
            'ksize': [1, 8, 1],
            'stride': [1, 8, 1]
        })
        self.pool.append({
            'ksize': [1, 8, 1],
            'stride': [1, 8, 1]
        })
        self.pool.append({
            'ksize': [1, 4, 1],
            'stride': [1, 4, 1]
        })
        self.pool.append({
            'ksize': [1, 4, 1],
            'stride': [1, 4, 1]
        })
        self.pool.append({
            'ksize': [1, 4, 1],
            'stride': [1, 4, 1]
        })

    def set_is_training(self, is_training):
        for i in self.conv:
            i['is_training'] = is_training
        for i in self.NN:
            i['is_training'] = is_training