import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal

mpl.rcParams['font.size'] = 10.
mpl.rcParams['font.family'] = 'Comic Sans MS'
# mpl.rcParams['axes.labesize'] = 8.
# mpl.rcParams['xtick.labelsize'] = 6.
# mpl.rcParams['ytick.labesize'] = 6.



def test_2d():
    i = -1
    j = 0
    while 1:
        i = i + 1
        data_i = np.load('/home/pkushi/dataset_2d4s1c/noise/%s.npy'%i)
        data_i[0,:] = (data_i[0,:]-data_i[0,:].min())/(data_i[0,:].max()-data_i[0,:].min())*2-1
        data_i[1,:] = (data_i[1,:]-data_i[1,:].min())/(data_i[1,:].max()-data_i[1,:].min())*2-1
        data_i[2,:] = (data_i[2,:]-data_i[2,:].min())/(data_i[2,:].max()-data_i[2,:].min())*2-1
        data_i[3,:] = (data_i[3,:]-data_i[3,:].min())/(data_i[3,:].max()-data_i[3,:].min())*2-1
        fig = plt.figure()
        l = 400*100
        plt.plot(range(l), data_i[0, :], label='0')
        plt.plot(range(l), data_i[1, :]+3, label='1')
        plt.plot(range(l), data_i[2, :]+6, label='2')
        plt.plot(range(l), data_i[3, :]+9, label='3')
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        plt.show()

        # if data_i.shape[0] != 4 or data_i.shape[1] != 40000:
        #     print(i)
        #     print(data_i.shape)
        # if i%400==0:
        #     print('%02d%%' % (i /400))
    # print(j)


def test_1d1s3c():
    i = -1
    j = 0
    while 1:
        i = i+1
        data_i = np.load('/home/pkushi/dataset_1d1s3c/events/%s.npy' % i).T
        data_i[0, :] = (data_i[0, :] - data_i[0, :].min()) / (data_i[0, :].max() - data_i[0, :].min()) * 2 - 1
        data_i[1, :] = (data_i[1, :] - data_i[1, :].min()) / (data_i[1, :].max() - data_i[1, :].min()) * 2 - 1
        data_i[2, :] = (data_i[2, :] - data_i[2, :].min()) / (data_i[2, :].max() - data_i[2, :].min()) * 2 - 1

        # data_i[0, :] = band_pass(data_i[0, :], 0.5, 5)
        # data_i[1, :] = band_pass(data_i[1, :], 0.5, 5)
        # data_i[2, :] = band_pass(data_i[2, :], 0.5, 5)
        fig = plt.figure()
        l = 200 * 100
        plt.plot(np.linspace(0,200,l), data_i[0, :]+4, color='r', label='E')
        plt.plot(np.linspace(0,200,l), data_i[1, :] + 2, color='g', label='N')
        plt.plot(np.linspace(0,200,l), data_i[2, :], color='b', label='U')

        plt.title('a EQ sample in SK')
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        plt.xlabel('time(s)')
        plt.yticks([])
        plt.show()

        # if data_i.shape[0] != 4 or data_i.shape[1] != 40000:
        #     print(i)
        #     print(data_i.shape)
        # if i%400==0:
        #     print('%02d%%' % (i /400))
    # print(j)


def test_1d3s3c():
    i = -1
    j = 0
    while 1:
        i = i+1
        data_i = np.load('/home/pkushi/dataset_1d3s3c/tremors/%s.npy' % i).T
        data_i[0, :] = (data_i[0, :] - data_i[0, :].min()) / (data_i[0, :].max() - data_i[0, :].min()) * 2 - 1
        data_i[1, :] = (data_i[1, :] - data_i[1, :].min()) / (data_i[1, :].max() - data_i[1, :].min()) * 2 - 1
        data_i[2, :] = (data_i[2, :] - data_i[2, :].min()) / (data_i[2, :].max() - data_i[2, :].min()) * 2 - 1
        data_i[3, :] = (data_i[3, :] - data_i[3, :].min()) / (data_i[3, :].max() - data_i[3, :].min()) * 2 - 1
        data_i[4, :] = (data_i[4, :] - data_i[4, :].min()) / (data_i[4, :].max() - data_i[4, :].min()) * 2 - 1
        data_i[5, :] = (data_i[5, :] - data_i[5, :].min()) / (data_i[5, :].max() - data_i[5, :].min()) * 2 - 1
        data_i[6, :] = (data_i[6, :] - data_i[6, :].min()) / (data_i[6, :].max() - data_i[6, :].min()) * 2 - 1
        data_i[7, :] = (data_i[7, :] - data_i[7, :].min()) / (data_i[7, :].max() - data_i[7, :].min()) * 2 - 1
        data_i[8, :] = (data_i[8, :] - data_i[8, :].min()) / (data_i[8, :].max() - data_i[8, :].min()) * 2 - 1
        fig = plt.figure()
        l = 200 * 100
        plt.plot(range(l), data_i[0, :], label='1sE')
        plt.plot(range(l), data_i[1, :] + 3, label='1sN')
        plt.plot(range(l), data_i[2, :] + 6, label='1sU')
        plt.plot(range(l), data_i[3, :] + 9, label='2sE')
        plt.plot(range(l), data_i[4, :] + 12, label='2sN')
        plt.plot(range(l), data_i[5, :] + 15, label='2sU')
        plt.plot(range(l), data_i[6, :] + 18, label='3sE')
        plt.plot(range(l), data_i[7, :] + 21, label='3sN')
        plt.plot(range(l), data_i[8, :] + 24, label='3sU')
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        plt.show()

        # if data_i.shape[0] != 4 or data_i.shape[1] != 40000:
        #     print(i)
        #     print(data_i.shape)
        # if i%400==0:
        #     print('%02d%%' % (i /400))
    # print(j)


def test_T_1d1s3c(area):
    i = -1
    j = 0
    while 1:
        i = i+1
        data_i = np.load('/home/pkushi/dataset_T_1d1s3c_test/noise_aichi/%s.npy' % i).T
        data_i[0, :] = (data_i[0, :] - data_i[0, :].min()) / (data_i[0, :].max() - data_i[0, :].min()) * 2 - 1
        data_i[1, :] = (data_i[1, :] - data_i[1, :].min()) / (data_i[1, :].max() - data_i[1, :].min()) * 2 - 1
        data_i[2, :] = (data_i[2, :] - data_i[2, :].min()) / (data_i[2, :].max() - data_i[2, :].min()) * 2 - 1
        fig = plt.figure()
        l = 200 * 100
        plt.plot(np.linspace(0, 200, l), data_i[0, :] + 4, color='r', label='E')
        plt.plot(np.linspace(0, 200, l), data_i[1, :] + 2, color='g', label='N')
        plt.plot(np.linspace(0, 200, l), data_i[2, :], color='b', label='U')
        plt.xlabel('time(s)')
        plt.yticks([])
        plt.title('a %s tremor sample'%area)
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        plt.show()

        # if data_i.shape[0] != 4 or data_i.shape[1] != 40000:
        #     print(i)
        #     print(data_i.shape)
        # if i%400==0:
        #     print('%02d%%' % (i /400))
    # print(j)
def band_pass(wave, s, e, interal=0.01):
    nyq_freq = 1/2/interal
    low = s/nyq_freq
    high = e/nyq_freq
    b,a = signal.butter(5, Wn=[low, high], btype='bandpass')

    result = signal.lfilter(b, a, wave)

    return result

test_T_1d1s3c('aichi')
# test_1d1s3c()