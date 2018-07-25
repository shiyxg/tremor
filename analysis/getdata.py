import numpy as np
import obspy as ob
import os
import matplotlib.pyplot as plt
import datetime

DATA_PATH = r'/media/pkushi/86847781-60d2-4f71-8ed9-24e2b48d79e1/sac'
PATH_CHAR = '/'


class Wave:
    '''
    this is a class to get waveform data
    need the sac file's path, there are some problems when this file was named as wave.py
    '''
    def __init__(self, path=DATA_PATH, path_char=PATH_CHAR):
        self.path = path
        self.path_char = path_char
        self.dir = os.listdir(path)

    def cut(self, data_path, start, end):
        '''
        to return a wave according the sequence in data_path, all data in middle will be picked
        :param data_path: a list about the file ready to pick all 1 h record expect the start and end one
        :param start: the start file in data_path's start, second(0.01*x, x is 1,2 3..)
        :param end: the end file in data_path's end, second(0.01*y, y is 1,2,3...)
        :return:
        '''
        data = []
        for i in range(len(data_path)):
            # in the first data(hour)
            if i == 0:
                start_i = int(start*100)
            else:
                start_i = 0
            # in the last data(hour)
            if i == len(data_path)-1:
                end_i = int(end*100)
            else:
                end_i = None
            try:
                wave_data = ob.read(data_path[i])[0].data
            except IOError:
                # if I can't read the datafile, replace it with random in [0,1)
                wave_data = np.random.random(360000)
            # the datafile miss a time cut, replace it with random in [0,1)
            if len(wave_data)<360000:
                wave_data = np.random.random(360000)
            # print(wave_data.shape)
            # print(start_i)
            # print(end_i)
            data.extend(wave_data[start_i:end_i])
        return data

    def get_path(self, date, hour_s, hour_e, station):
        '''
        according the datatime and station . return file's path
        the data should in a same date
        :param date: about date. '170401'
        :param hour_s: the start time 19
        :param hour_e:the end time 23
        :param station: 'N.QWER'
        :return: pathes about the files needed to be read
        '''
        p = self.path
        c = self.path_char
        hours = np.linspace(hour_s, hour_e, hour_e - hour_s + 1)
        # get different's components' path
        U = []
        E = []
        N = []
        for i in hours:
            U.append(p + c + date + c + date + '%02d' % i + '00' + '_' + station + '_U.s')
            E.append(p + c + date + c + date + '%02d' % i + '00' + '_' + station + '_E.s')
            N.append(p + c + date + c + date + '%02d' % i + '00' + '_' + station + '_N.s')

        data_path = [E, N, U]
        # E = []
        # for i in hours:
        #     E.append(p + c + date + c + date + '%02d' % i + '00' + '_' + station + '_E.s')
        #
        # data_path = [E]
        return data_path

    def cut_wave(self, start, end, station):
        # duration should shorter than 1d
        if (start['date'] not in self.dir) or (end['date'] not in self.dir):
            print(start['date'])
            raise ValueError('we have no wave data in this date')

        # 起始于终止均在同一日期
        if start['date'] == end['date']:
            date = start['date']
            hour_s = int(start['hour'])
            hour_e = int(end['hour'])
            sec_s = start['sec']
            sec_e = end['sec']
            # get the files path
            data_path = self.get_path(date, hour_s, hour_e, station)
            # print(start)
            # print(end)
            data = []
            for i in data_path:
                # get different components' wave
                component = self.cut(i, sec_s, sec_e)
                data.append(np.array(component))
        # 跨日期
        else:
            sec_s = start['sec']
            sec_e = end['sec']
            date_s = start['date']
            date_e = end['date']

            # the first date's data
            date = date_s
            hour_s = int(start['hour'])
            hour_e = 23
            data_path1 = self.get_path(date, hour_s, hour_e, station)
            # print(data_path1)
            # the second date' data
            date = date_e
            hour_s = 0
            hour_e = int(end['hour'])
            data_path2 = self.get_path(date, hour_s, hour_e, station)
            # print(data_path2)
            # merge two days' data to continual data.
            data_path = data_path1
            data_path[0].extend(data_path2[0])
            data_path[1].extend(data_path2[1])
            data_path[2].extend(data_path2[2])

            data = []
            for i in data_path:
                component = self.cut(i, sec_s, sec_e)
                data.append(np.array(component))

        # print(data_path)
        return data

    def get_waveform(self, start, duration, station, shift):
        '''
        process the datetime question.
        by event_start and shift, get the start time; by duration to get the end time.
        then by station, get the waveform
        :param start: the event's start date and time
        :param duration: length of windows, second
        :param station: ..
        :param shift: the time to arrival at the station (s)
        :return: NED components result list
        '''
        date = int(start['date'])
        Y = date//10000+2000
        M = (date%10000)//100
        D = date%100
        hour = int(start['hour'])
        min = int(start['sec']//60)
        sec = int(start['sec']%60)
        microsec = int(start['sec']%1*1e6)

        # print([Y, M, D, hour, min, sec, microsec])
        time_event = datetime.datetime(Y, M, D, hour, min, sec, microsec)
        shift = datetime.timedelta(seconds=shift)
        duration = datetime.timedelta(seconds=duration)
        time_start = time_event+shift
        time_end = time_start + duration

        start = {
            'date': '%02d%02d%02d' % (time_start.year-2000,time_start.month, time_start.day),
            'hour': '%02d' % time_start.hour,
            'sec': time_start.minute*60 + time_start.second + time_start.microsecond*1e-6
        }
        end = {
            'date': '%02d%02d%02d' % (time_end.year - 2000, time_end.month, time_end.day),
            'hour': '%02d' % time_end.hour,
            'sec': time_end.minute * 60 + time_end.second + time_end.microsecond * 1e-6
        }

        result = self.cut_wave(start, end, station)
        return result



# a = Wave()
# b = a.get_waveform({'date':'170401', 'hour':'19', 'sec':200}, 10000,'N.AIOH', 3.5656)
# print(len(b[0]))
# plt.plot(range(10000),b[0])
# plt.show()

# start = {'date': '171023', 'hour': '00', 'sec': 170.19653955928553}
# station = 'N.GHKH'
# a = Wave()
# b = a.get_waveform(start=start, station=station, duration=400,shift=0)[0]
# print(len(b))