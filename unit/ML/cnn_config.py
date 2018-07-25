# -*- coding:utf-8


class CNNConf:
    def __init__(self):
        self.INPUT_SHAPE = [32, 32, 1]
        self.OUTPUT_SHAPE = [32, 32, 1]

        self.conv = []
        self.conv.append({
            'name': 'convBlock1',
            'times': 2,
            'filterSize': [3, 3],
            'outputChn': 64,
            'strideSize': [1, 1],
            'is_training': True
        })
        self.conv.append({
            'name': 'convBlock2',
            'times': 2,
            'filterSize': [3, 3],
            'outputChn': 128,
            'strideSize': [1, 1],
            'is_training': True
        })
        self.conv.append({
            'name': 'convBlock3',
            'times': 2,
            'filterSize': [3, 3],
            'outputChn': 256,
            'strideSize': [1, 1],
            'is_training': True
        })
        self.conv.append({
            'name': 'convBlock4',
            'times': 2,
            'filterSize': [3, 3],
            'outputChn': 512,
            'strideSize': [1, 1],
            'is_training': True
        })

        self.NN = []
        self.NN.append({
            'name': 'NN1',
            'num': 4096
        })
        self.NN.append({
            'name': 'NN2',
            'num': 4096
        })
        self.NN.append({
            'name': 'output',
            'num': self.INPUT_SHAPE[0]*self.INPUT_SHAPE[1]
        })


