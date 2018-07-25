import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import signal

NUM=1


def plot_2d4s1c(data, logits, step, pwd):
    wave = data[0]
    label = data[1].argmax(1)
    result = logits.argmax(1)
    for i in range(NUM):
        label_i = label[i]
        result_i = result[i]

        if result_i==label_i:
            identify = 'correct'
        else:
            identify = 'error'

        if label_i == 0:
            label_i = 'tremor'
        elif label_i == 2:
            label_i = 'noise'
        else:
            label_i = 'EQ'

        if result_i == 0:
            result_i = 'tremor'
        elif result_i == 2:
            result_i = 'noise'
        else:
            result_i = 'EQ'
        if (result_i=='tremor' or label_i=='tremor') or\
                (label_i=='noise') \
                and identify=='error':
            wave[i, 0, :, 0] = band_pass(wave[i, 0, :, 0], 1, 10)
            wave[i, 1, :, 0] = band_pass(wave[i, 1, :, 0], 1, 10)
            wave[i, 2, :, 0] = band_pass(wave[i, 2, :, 0], 1, 10)
            wave[i, 3, :, 0] = band_pass(wave[i, 3, :, 0], 1, 10)

        plt.figure()
        plt.plot(np.linspace(0, 400, 40000), wave[i, 0, :, 0] + 6, color='r', label='1sE')
        plt.plot(np.linspace(0, 400, 40000), wave[i, 1, :, 0] + 4, color='g', label='2sE')
        plt.plot(np.linspace(0, 400, 40000), wave[i, 2, :, 0] + 2, color='b', label='3sE')
        plt.plot(np.linspace(0, 400, 40000), wave[i, 3, :, 0] + 0, color='k', label='4sE')

        print('identify result:'+identify)
        plt.title('label:' + label_i + ',result:' + result_i)
        plt.xlabel('time(s)')
        plt.yticks([])
        plt.legend(bbox_to_anchor=(0.95, 1), loc=2, borderaxespad=0.)
        plt.savefig(pwd + '/'+identify+'_%s_%s.png' % (step, i))

        plt.close('all')


def plot_1d1s3c(data, logits, step, pwd):

    wave = data[0]
    label = data[1].argmax(1)
    result = logits.argmax(1)
    for i in range(NUM):
        label_i = label[i]
        result_i = result[i]

        if result_i == label_i:
            identify = 'correct'
        else:
            identify = 'error'

        if label_i == 0:
            label_i = 'tremor'
        elif label_i == 2:
            label_i = 'noise'
        else:
            label_i = 'EQ'

        if result_i == 0:
            result_i = 'tremor'
        elif result_i == 2:
            result_i = 'noise'
        else:
            result_i = 'EQ'

        if (result_i=='tremor' or label_i=='tremor') and identify=='error':
            print(1)
            wave[i, :, 0] = band_pass(wave[i, :, 0], 1, 10)
            wave[i, :, 1] = band_pass(wave[i, :, 1], 1, 10)
            wave[i, :, 2] = band_pass(wave[i, :, 2], 1, 10)

        plt.figure()
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 0] + 4, color='r', label='E')
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 1] + 2, color='g', label='N')
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 2] + 0, color='b', label='U')

        # print('identify result:' + identify)
        plt.title('label:' + label_i + ',result:' + result_i)
        plt.xlabel('time(s)')
        plt.yticks([])
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        # if identify=='error':
        plt.savefig(pwd + '/' + '%s_s_%s_r_%s_%s.png' % (step, label_i, result_i, i))

        plt.close('all')


def plot_1d3s3c(data, logits, step, pwd):
    wave = data[0]
    label = data[1].argmax(1)
    result = logits.argmax(1)
    for i in range(NUM):
        label_i = label[i]
        result_i = result[i]

        if result_i == label_i:
            identify = 'correct'
        else:
            identify = 'error'

        if label_i == 0:
            label_i = 'tremor'
        elif label_i == 2:
            label_i = 'noise'
        else:
            label_i = 'EQ'

        if result_i == 0:
            result_i = 'tremor'
        elif result_i == 2:
            result_i = 'noise'
        else:
            result_i = 'EQ'

        if (result_i=='tremor' or label_i=='tremor') and identify=='error':
            wave[i, :, 0] = band_pass(wave[i, :, 0], 1, 10)
            wave[i, :, 1] = band_pass(wave[i, :, 1], 1, 10)
            wave[i, :, 2] = band_pass(wave[i, :, 2], 1, 10)
            wave[i, :, 3] = band_pass(wave[i, :, 3], 1, 10)
            wave[i, :, 4] = band_pass(wave[i, :, 4], 1, 10)
            wave[i, :, 5] = band_pass(wave[i, :, 5], 1, 10)
            wave[i, :, 6] = band_pass(wave[i, :, 6], 1, 10)
            wave[i, :, 7] = band_pass(wave[i, :, 7], 1, 10)
            wave[i, :, 8] = band_pass(wave[i, :, 8], 1, 10)
        plt.figure()
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 0] + 16, color='r', label='1sE')
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 1] + 14, color='g', label='1sN')
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 2] + 12, color='b', label='1sU')
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 3] + 10, color='r', label='2sE')
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 4] + 8, color='g', label='2sN')
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 5] + 6, color='b', label='2sU')
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 6] + 4, color='r', label='3sE')
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 7] + 2, color='g', label='3sN')
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 8] + 0, color='b', label='3sU')

        plt.xlabel('time(s)')
        plt.yticks([])
        plt.legend(bbox_to_anchor=(0.95, 1), loc=2, borderaxespad=0.)

        print('identify result:' + identify)
        plt.title('label:' + label_i + ',result:' + result_i)
        if identify=='error':
            plt.savefig(pwd + '/' + identify + '_%s_%s.png' % (step, i))

        plt.close('all')


def band_pass(wave, s, e, interal=0.01):
    nyq_freq = 1/2/interal
    low = s/nyq_freq
    high = e/nyq_freq
    b,a = signal.butter(5, Wn=[low, high], btype='bandpass')

    result = signal.lfilter(b, a, wave)

    return result


# x = np.load('/home/pkushi/dataset_1d1s3c/tremors/3.npy')[:,1]
# x = (x-x.min())/(x.max()-x.min())*2-1
# x_filter = band_pass(x, 1, 10)
#
# plt.plot(np.linspace(0,200, len(x)), x, label='befor filter')
# plt.plot(np.linspace(0,200, len(x)), x_filter+3, label='after filter')
# plt.legend()
#
# plt.show()


def plot_T_1d1s3c(data, logits, step, pwd):

    wave = data[0]
    label = data[1].argmax(1)
    result = logits.argmax(1)
    for i in range(NUM):
        label_i = label[i]
        result_i = result[i]

        if result_i == label_i:
            identify = 'correct'
        else:
            identify = 'error'

        if label_i == 0:
            label_i = 'sg'
        elif label_i == 2:
            label_i = 'kii'
        else:
            label_i = 'aichi'

        if result_i == 0:
            result_i = 'sg'
        elif result_i == 2:
            result_i = 'kii'
        else:
            result_i = 'aichi'

        wave[i, :, 0] = band_pass(wave[i, :, 0], 1, 10)
        wave[i, :, 1] = band_pass(wave[i, :, 1], 1, 10)
        wave[i, :, 2] = band_pass(wave[i, :, 2], 1, 10)

        plt.figure()
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 0] + 4, color='r', label='E')
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 1] + 2, color='g', label='N')
        plt.plot(np.linspace(0, 200, 20000), wave[i, :, 2] + 0, color='b', label='U')

        # print('identify result:' + identify)
        plt.title('label:' + label_i + ',result:' + result_i)
        plt.xlabel('time(s)')
        plt.yticks([])
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        print('identify result:' + identify)
        plt.title('label:' + label_i + ',result:' + result_i)
        plt.savefig(pwd + '/' +  '%s_s_%s_r_%s_%s.png' % (step, label_i, result_i, i))

        plt.close('all')


def plot_features(features, pwd):

    for j in [2,3]:
        f = features[j]
        f = f[0,:,:]
        if f.shape[1]<200:
            plt.figure(figsize=[10, f.shape[1]*0.6])
            for i in range(f.shape[1]):
                plt.plot(np.linspace(0,200, f.shape[0]), f[:,i]/(f[:,i].max())+i, label=i)
                plt.xlabel('time(s)')
        else:

            # for i in range(f.shape[1]):
            #     f[:,i] = f[:,i]/(f[:,i].max())

            f = f.T
            plt.figure(figsize=[f.shape[1]*0.2, f.shape[0]*0.2])
            plt.imshow(f, cmap=cm.seismic)
            # plt.colorbar()

        # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        plt.savefig(pwd+'_%s.png'%j)
        plt.close('all')



