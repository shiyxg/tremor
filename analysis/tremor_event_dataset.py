import numpy as np
import random

from analysis.getdata import Wave
from analysis.earthquake import Event
PATH = '/home/pkushi/dataset/'
SAMPLING_RATE = 100

duration = 400
gap = 2
shuffle = 1
chn = 4

wave_data = Wave()

# get Tremors
# tremors = tremor(chn=chn).tremor
# tremor_sample = []
# num = 0
# for i in range(len(tremors)):
# # for i in [84,85,86,87,88,89,90,91,92,93,94]:
#     for j in range(gap):
#         start_e = tremors[i]
#         shift = -1 * (duration/gap * (j + random.random()))
#         # print(tremors[i])
#         # print(shift)
#         wave_sample = []
#         for station in tremors[i]['station']:
#             wave_sample_s = wave_data.get_waveform(start=start_e, station=station[0], shift=shift, duration=duration)[0]
#             wave_sample.append(wave_sample_s)
#             if len(wave_sample_s) != duration*SAMPLING_RATE:
#                 print(tremors[i])
#                 print(len(wave_sample_s))
#                 raise ValueError('finded')
#         for k in range(shuffle):
#             random.shuffle(wave_sample)
#             sample = np.array(wave_sample)
#             np.save(PATH+'tremors/%s.npy'%num, sample)
#             num = num + 1
#
#     print(i)

events = Event(chn=chn).event

event_sample = []
num = 0
for i in range(len(events)):
    # for i in [84,85,86,87,88,89,90,91,92,93,94]:
    for j in range(gap):
        start_e = events[i]
        shift = -1 * (duration/gap * (j + random.random()))
        # print(tremors[i])
        # print(shift)
        wave_sample = []
        for station in events[i]['station']:
            wave_sample_s = wave_data.get_waveform(start=start_e, station=station[0], shift=shift, duration=duration)[0]
            wave_sample.append(wave_sample_s)

        for k in range(shuffle):
            random.shuffle(wave_sample)
            sample = np.array(wave_sample)
            np.save(PATH+'events/%s.npy'%num, sample)
            num = num + 1

    if i%100 == 0:
        print('%s%%'%(i/len(events)*100))