# -*- coding: utf-8


class FCNConf1:
    def __init__(self, is_training=None):
        self.INPUT_SHAPE = [250, 400, 1]
        self.OUTPUT_SHAPE = [250, 400, 1]

        self.conv1 = {
            'name'              :'convBlock1',
            'times'             :2,
            'filterSize'        :[3, 3],
            'outputChn'         :64,
            'strideSize'        :[1, 1],
            'is_training'       :True
        }
        self.conv2 = {
            'name'              :'convBlock2',
            'times'             :2,
            'filterSize'        :[3, 3],
            'outputChn'         :128,
            'strideSize'        :[1, 1],
            'is_training'       :True
        }
        self.conv3 = {
            'name'              :'convBlock3',
            'times'             :2,
            'filterSize'        :[3, 3],
            'outputChn'         :256,
            'strideSize'        :[1, 1],
            'is_training'       :True
        }
        self.conv4 = {
            'name'              :'convBlock4',
            'times'             :2,
            'filterSize'        :[3, 3],
            'outputChn'         :512,
            'strideSize'        :[1, 1],
            'is_training'       :True
        }

        self.deconv1 = {
            'name'              :'deconv1',
            'scale'             :1,
            'outputChn'         :1,
            'outputSize'        :self.OUTPUT_SHAPE,
            'is_training'       :True
        }
        self.deconv2 = {
            'name'              :'deconv2',
            'scale'             :2,
            'outputChn'         :1,
            'outputSize'        :self.OUTPUT_SHAPE,
            'is_training'       :True
        }
        self.deconv3 = {
            'name'              :'deconv3',
            'scale'             :4,
            'outputChn'         :1,
            'outputSize'        :self.OUTPUT_SHAPE,
            'is_training'       :True
        }
        self.deconv4 = {
            'name'              :'deconv4',
            'scale'             :8,
            'outputChn'         :1,
            'outputSize'        :self.OUTPUT_SHAPE,
            'is_training'       :True
        }

        self.fuse = {
                'name': 'convTo1',
                'times': 1,
                'filterSize': [1, 1],
                'outputChn':  self.OUTPUT_SHAPE[2],
                'strideSize': [1, 1],
                'is_training': True
            }

        self.conv = [self.conv1,self.conv2, self.conv3, self.conv4]
        self.deconv = [self.deconv1,self.deconv2, self.deconv2, self.deconv2]
        if is_training is not None:
            for i in self.conv:
                i['is_training'] = is_training
            for i in self.deconv:
                i['is_training'] = is_training
            self.fuse['is_training'] = is_training
