import numpy as np
import random
from analysis.tremor import Tremor
from analysis.getdata import Wave
from analysis.earthquake import Event

DATA_PATH={'sg': r'/media/pkushi/86847781-60d2-4f71-8ed9-24e2b48d79e1/sac',
           'kii': '/media/pkushi/HDPH-UT/kii',
           'aichi': '/media/pkushi/HDPH-UT/aichi'}

SAMPLING_RATE = 100
duration = 200
gap = 2
shuffle = 1
chn = 4
AREA = 'sg'
PATH = '/home/pkushi/dataset_T_1d1s3c/'+AREA

wave_data = Wave(DATA_PATH[AREA])
# get Tremors
tremors = Tremor(chn=chn, area=AREA).tremor
print(len(tremors))
# get EQ
#EQ = Event(chn=chn).event
event_kind = tremors
num = 0
for i in range(len(event_kind)):
# for i in [84,85,86,87,88,89,90,91,92,93,94]:
    for j in range(gap):
        start_e = event_kind[i]
        shift = -1 * (duration/gap * (j + random.random()))
        # print(tremors[i])
        # print(shift)
        wave_sample = []
        for station in event_kind[i]['station']:
            wave_sample_s = wave_data.get_waveform(start=start_e, station=station[0], shift=shift, duration=duration)
            wave_sample.append(wave_sample_s)
            if len(wave_sample_s) != 3 or \
                    len(wave_sample_s[0]) != duration*SAMPLING_RATE or \
                    len(wave_sample_s[1]) != duration*SAMPLING_RATE or \
                    len(wave_sample_s[2]) != duration*SAMPLING_RATE:
                print(event_kind[i])
                print(len(wave_sample_s))
                raise ValueError('finded')

            sample = np.array(wave_sample_s)
            sample = sample.reshape([3, duration * SAMPLING_RATE]).T
            np.save(PATH + '/%s.npy' % num, sample)
            num = num + 1
        # sample = np.array(wave_sample)
        # sample = sample.reshape([chn*3, duration*SAMPLING_RATE]).T
        # np.save(PATH + '/%s.npy' % num, sample)
        # num = num + 1
        # for k in range(shuffle):
        #     random.shuffle(wave_sample)
        #     sample = np.array(wave_sample)
        #     np.save(PATH+'tremors/%s.npy'%num, sample)
        #     num = num + 1
    print(i)

# events = Event(chn=chn).event
#
# event_sample = []
# num = 0
# for i in range(len(events)):
# # for i in [84,85,86,87,88,89,90,91,92,93,94]:
#     for j in range(gap):
#         start_e = events[i]
#         shift = -1 * (duration/gap * (j + random.random()))
#         # print(tremors[i])
#         # print(shift)
#         wave_sample = []
#         for station in events[i]['station']:
#             wave_sample_s = wave_data.get_waveform(start=start_e, station=station[0], shift=shift, duration=duration)[0]
#             wave_sample.append(wave_sample_s)
#             if len(wave_sample_s) != duration*SAMPLING_RATE:
#                 print(events[i])
#                 print(station)
#                 print(len(wave_sample_s))
#                 raise ValueError('finded')
#         for k in range(shuffle):
#             random.shuffle(wave_sample)
#             sample = np.array(wave_sample)
#             np.save(PATH+'events_4000/%s.npy'%num, sample)
#             num = num + 1
#
#     if i%100 == 0:
#         print('%s%%'%(i/len(events)*100))