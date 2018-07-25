import os
import numpy as np
import random
from analysis.station import Station
from analysis.getdata import Wave

# consider noise
def is_in_area(area, log):
    result = True
    log = [1,2,3,4,5,6,log[0], log[1]]
    if area == 'sg':
        if (log[6] > 134.5) or (log[6] < 131.8):
            result = False
        if (log[7] > 34.5) or (log[7] < 32.5):
            result = False

    if area == 'kii':
        if (log[6] > 136.8) or (log[6] < 134.8):
            result = False
        if (log[7] > 35) or (log[7] < 33.4):
            result = False

    if area == 'aichi':
        if (log[6] > 138.4) or (log[6] < 136.8):
            result = False
        if (log[7] > 35.6) or (log[7] < 34.6):
            result = False
    return result

area = 'kii'
DATA_PATH={'sg': r'/media/pkushi/86847781-60d2-4f71-8ed9-24e2b48d79e1/sac',
           'kii': '/media/pkushi/HDPH-UT/kii',
           'aichi': '/media/pkushi/HDPH-UT/aichi'}
PATH = '/home/pkushi/dataset_T_1d1s3c_test/noise_kii'
data_path = '/media/pkushi/HDPH-UT/'+area
SAMPLING_RATE = 100
sample_num = 1000
duration = 200
shuffle = 1
chn = 1

stat = Station(AREA=area)
print(stat.inf)
wave = Wave(DATA_PATH[area])
stations = []
for i in stat.valid_station:
    lon_s, lat_s, _ = stat.inf[i]
    # disgard the station out of the area
    if not is_in_area(area, [lon_s, lat_s]):
        continue
    stations.append(i)

group = []

for i in stations:
    result = stat.get_stations(event_loc=stat.inf[i], radius=50, num=chn)
    if result is not None:
        c = []
        for j in result:
            c.append(j[0])
        group.append(c)

print(group)
all_date = os.listdir(data_path)
# all_date.remove('CONV.sh')
# all_date.remove('sakura.tbl')
# all_date.remove('WIN.list')
# all_date.remove('WIN.list.10')
# all_date.remove('WIN.list.11')
# all_date.remove('WIN.list.12')
# all_date.remove('WIN.list.org')
all_date.remove('CONV.sh')
all_date.remove('WIN.list')
all_date.remove('hinet.tbl')
# all_date.remove('xxx.sac3')

hours = range(24)
num = 0
for i in group:
    for j in range(sample_num//len(group)+len(group)):
        index_date = int(random.random()*len(all_date))
        index_hour = int()
        date = all_date[index_date]
        hour = '%02d'%(hours[index_hour])
        sec = random.random()*3600
        data = []
        for st in i:
            start = {
                'date': date,
                'hour': hour,
                'sec': sec,
            }

            wave_data = wave.get_waveform(start=start, duration=duration, station=st, shift=0)
            data.append(wave_data)
            if len(wave_data) != 3 or \
                    len(wave_data[0]) != duration * SAMPLING_RATE or \
                    len(wave_data[1]) != duration * SAMPLING_RATE or \
                    len(wave_data[2]) != duration * SAMPLING_RATE:
                print(start)
                print(st)
                print(len(wave_data))
                raise ValueError('length is not equal')

        data = np.array(data)
        data = data.reshape([chn*3, duration*SAMPLING_RATE])
        data = data.T
        np.save(PATH + '/%s.npy' % num, data)

        num = num + 1
        if num % 100 == 0:
            print('%02d%%'%(num/sample_num*100))





