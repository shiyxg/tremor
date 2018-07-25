import numpy as np
import os
DATASET_PATH = '/home/pkushi/dataset'


class Data_2d4s1c:
    def __init__(self, datapath = DATASET_PATH):
        self.tremor_path = datapath + '/tremors'
        self.event_path = datapath + '/events'
        self.noise_path = datapath + '/noise'

        self.tremor_num = len(os.listdir(self.tremor_path))
        self.event_num = len(os.listdir(self.event_path))
        self.noise_num = len(os.listdir(self.noise_path))

        example = np.load(self.tremor_path+'/0.npy')
        self.shape = example.shape

    def next_batch(self, num):
        data = np.zeros([num, self.shape[0], self.shape[1], 1])
        label = np.zeros([num, 3])
        kind = np.random.random(num)*100
        index = np.random.random(num)

        # kind = [40] * (num // 2) + [60] * (num - num // 2)
        # index = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        for i in range(num):
            if kind[i] < 33:
                path = self.tremor_path
                index_i = int(index[i]*self.tremor_num)
                label_i = [1.0, 0., 0.]
            elif kind[i] >= 33 and (kind[i] < 66):
                path = self.event_path
                index_i = int(index[i] * self.event_num)
                label_i = [0., 1.0, 0.0]
            else:
                path = self.noise_path
                index_i = int(index[i] * self.noise_num)
                label_i = [0., 0., 1.0]
            data_i = np.load(path+'/%s.npy'%index_i)
            data_i[0, :] = (data_i[0, :] - data_i[0, :].min()) / (data_i[0, :].max() - data_i[0, :].min()) * 2 - 1
            data_i[1, :] = (data_i[1, :] - data_i[1, :].min()) / (data_i[1, :].max() - data_i[1, :].min()) * 2 - 1
            data_i[2, :] = (data_i[2, :] - data_i[2, :].min()) / (data_i[2, :].max() - data_i[2, :].min()) * 2 - 1
            data_i[3, :] = (data_i[3, :] - data_i[3, :].min()) / (data_i[3, :].max() - data_i[3, :].min()) * 2 - 1
            data_i = data_i-data_i.mean()
            data_i = data_i/data_i.max()
            data[i, :, :, 0] = data_i
            label[i, :] = label_i

        return [data, label]

    def get_tremors(self, num):
        data = np.zeros([num, self.shape[0], self.shape[1], 1])
        label = np.zeros([num, 3])
        # kind = np.random.random(num)*100
        # index = np.random.random(num)
        index = np.random.random(num)
        for i in range(num):
            path = self.tremor_path
            index_i = int(index[i] * self.tremor_num)
            label_i = [1.0, 0., 0.]
            data_i = np.load(path + '/%s.npy' % index_i)
            data_i[0, :] = (data_i[0, :] - data_i[0, :].min()) / (data_i[0, :].max() - data_i[0, :].min()) * 2 - 1
            data_i[1, :] = (data_i[1, :] - data_i[1, :].min()) / (data_i[1, :].max() - data_i[1, :].min()) * 2 - 1
            data_i[2, :] = (data_i[2, :] - data_i[2, :].min()) / (data_i[2, :].max() - data_i[2, :].min()) * 2 - 1
            data_i[3, :] = (data_i[3, :] - data_i[3, :].min()) / (data_i[3, :].max() - data_i[3, :].min()) * 2 - 1

            data[i, :, :, 0] = data_i
            label[i, :] = label_i

        return [data, label]

    def get_samples(self, num):
        data = np.zeros([num, self.shape[0], self.shape[1], 1])
        label = np.zeros([num, 3])

        kind = np.random.random(num)*100
        index = np.random.random(num)
        for i in range(num):
            if kind[i] < 50:
                path = self.tremor_path
                index_i = int(index[i] * self.tremor_num)
                label_i = [1.0, 0., 0.]
            elif kind[i] >= 200000 and (kind[i] < 40):
                path = self.event_path
                index_i = int(index[i] * self.event_num)
                label_i = [0., 1.0, 0.0]
            else:
                path = self.noise_path
                index_i = int(index[i] * self.noise_num)
                label_i = [0., 0., 1.0]
            data_i = np.load(path + '/%s.npy' % index_i)
            data_i[0, :] = (data_i[0, :] - data_i[0, :].min()) / (data_i[0, :].max() - data_i[0, :].min()) * 2 - 1
            data_i[1, :] = (data_i[1, :] - data_i[1, :].min()) / (data_i[1, :].max() - data_i[1, :].min()) * 2 - 1
            data_i[2, :] = (data_i[2, :] - data_i[2, :].min()) / (data_i[2, :].max() - data_i[2, :].min()) * 2 - 1
            data_i[3, :] = (data_i[3, :] - data_i[3, :].min()) / (data_i[3, :].max() - data_i[3, :].min()) * 2 - 1

            data[i, :, :, 0] = data_i
            label[i, :] = label_i

        return [data, label]


class Data_1d1s3c:
    def __init__(self, datapath = DATASET_PATH):
        self.tremor_path = datapath + '/tremors'
        self.event_path = datapath + '/events'
        self.noise_path = datapath + '/noise'

        self.tremor_num = len(os.listdir(self.tremor_path))
        self.event_num = len(os.listdir(self.event_path))
        self.noise_num = len(os.listdir(self.noise_path))

        example = np.load(self.tremor_path+'/0.npy')
        self.shape = example.shape

    def next_batch(self, num):
        data = np.zeros([num, self.shape[0], self.shape[1]])
        label = np.zeros([num, 3])
        kind = np.random.random(num)*100
        index = np.random.random(num)

        # kind = [40] * (num // 2) + [60] * (num - num // 2)
        # index = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        for i in range(num):
            if kind[i] < 40:
                path = self.tremor_path
                index_i = int(index[i]*self.tremor_num)
                label_i = [1.0, 0., 0.]
            elif kind[i] >= 70:
                path = self.event_path
                index_i = int(index[i] * self.event_num)
                label_i = [0., 1.0, 0.0]
            else:
                path = self.noise_path
                index_i = int(index[i] * self.noise_num)
                label_i = [0., 0., 1.0]
            data_i = np.load(path+'/%s.npy'%index_i)
            # noise = int(self.noise_num * np.random.random())
            # data_i[:, 1] = np.load(self.noise_path + '/%s.npy'%noise)[:, 0]
            # data_i[:, 1] = np.sin(np.linspace(1, 2*np.pi*200*0.2, 20000))
            # data_i[:, 1] = np.zeros(20000)
            for j in range(self.shape[1]):
                data_i[:, j] = (data_i[:, j] - data_i[:, j].min()) / (data_i[:, j].max() - data_i[:, j].min()) * 2 - 1

            data[i, :, :] = data_i
            # data[i, :, 0] = np.random.random(20000)
            label[i, :] = label_i

        return [data, label]

class Data_1d3s3c:
    def __init__(self, datapath = DATASET_PATH):
        self.tremor_path = datapath + '/tremors'
        self.event_path = datapath + '/events'
        self.noise_path = datapath + '/noise'

        self.tremor_num = len(os.listdir(self.tremor_path))
        self.event_num = len(os.listdir(self.event_path))
        self.noise_num = len(os.listdir(self.noise_path))

        example = np.load(self.tremor_path+'/0.npy')
        self.shape = example.shape

    def next_batch(self, num):
        data = np.zeros([num, self.shape[0], self.shape[1]])
        label = np.zeros([num, 3])
        kind = np.random.random(num)*100
        index = np.random.random(num)

        # kind = [40] * (num // 2) + [60] * (num - num // 2)
        # index = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        for i in range(num):
            if kind[i] < 33:
                path = self.tremor_path
                index_i = int(index[i]*self.tremor_num)
                label_i = [1.0, 0., 0.]
            elif kind[i] >= 33 and (kind[i] < 66):
                path = self.event_path
                index_i = int(index[i] * self.event_num)
                label_i = [0., 1.0, 0.0]
            else:
                path = self.noise_path
                index_i = int(index[i] * self.noise_num)
                label_i = [0., 0., 1.0]
            data_i = np.load(path+'/%s.npy'%index_i)
            for j in range(self.shape[1]):
                data_i[:, j] = (data_i[:, j] - data_i[:, j].min()) / (data_i[:, j].max() - data_i[:, j].min()) * 2 - 1

            data_i = data_i.T
            data_i = data_i.reshape([3, 3, self.shape[0]])
            np.random.shuffle(data_i)
            data_i = data_i.reshape([9, self.shape[0]])
            data_i = data_i.T

            data[i, :, :] = data_i
            label[i, :] = label_i

        return [data, label]


class Data_T_1d1s3c:
    def __init__(self, datapath = DATASET_PATH):
        self.sg_path = datapath + '/sg'
        self.kii_path = datapath + '/kii'
        self.aichi_path = datapath + '/aichi'

        self.sg_num = len(os.listdir(self.sg_path))
        self.kii_num = len(os.listdir(self.kii_path))
        self.aichi_num = len(os.listdir(self.aichi_path))

        example = np.load(self.sg_path+'/0.npy')
        self.shape = example.shape

    def next_batch(self, num):
        data = np.zeros([num, self.shape[0], self.shape[1]])
        label = np.zeros([num, 3])
        kind = np.random.random(num)*100
        index = np.random.random(num)

        # kind = [40] * (num // 2) + [60] * (num - num // 2)
        # index = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        for i in range(num):
            if kind[i] < 33:
                path = self.sg_path
                index_i = int(index[i]*self.sg_num)
                # path = '/home/pkushi/dataset_T_1d1s3c_test/tremors'
                # index_i = int(index[i]*400)
                label_i = [1.0, 0., 0.]
            elif kind[i] >= 33 and (kind[i] < 66):
                path = self.kii_path
                index_i = int(index[i] * self.kii_num)
                label_i = [0., 1.0, 0.0]
            else:
                path = self.aichi_path
                index_i = int(index[i] * self.aichi_num)
                label_i = [0., 0., 1.0]
            # index_i = 10
            data_i = np.load(path+'/%s.npy'%index_i)
            # aichi = int(self.aichi_num * np.random.random())
            # data_i[:, 1] = np.load(self.aichi_path + '/%s.npy'%aichi)[:, 0]
            # data_i[:, 1] = np.sin(np.linspace(1, 2*np.pi*200*0.2, 20000))
            # data_i[:, 1] = np.zeros(20000)
            for j in range(self.shape[1]):
                data_i[:, j] = (data_i[:, j] - data_i[:, j].min()) / (data_i[:, j].max() - data_i[:, j].min()) * 2 - 1

            data[i, :, :] = data_i
            # data[i,:,0] = np.random.random(20000)
            # data[i,:,1] = np.random.random(20000)
            label[i, :] = label_i

        return [data, label]


class Data_simple:
    def __init__(self, datapath):
        self.path = datapath

        self.num = len(os.listdir(self.path))

        example = np.load(self.path+'/0.npy')
        self.shape = example.shape

    def next_batch(self, num):
        data = np.zeros([num, self.shape[0], self.shape[1]])
        label = np.zeros([num, 3])
        index = np.random.random(num)

        # kind = [40] * (num // 2) + [60] * (num - num // 2)
        # index = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        for i in range(num):

            index_i = int(index[i] * self.num)
            label_i = [1., 0., 0.]
            data_i = np.load(self.path+'/%s.npy'%index_i)
            for j in range(self.shape[1]):
                data_i[:, j] = (data_i[:, j] - data_i[:, j].min()) / (data_i[:, j].max() - data_i[:, j].min()) * 2 - 1

            data[i, :, :] = data_i
            label[i, :] = label_i

        return [data, label]
