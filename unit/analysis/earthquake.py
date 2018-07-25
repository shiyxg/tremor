import numpy as np
from analysis.station import Station

CATALOG = r'/media/pkushi/86847781-60d2-4f71-8ed9-24e2b48d79e1/data/jma1cat_flag.txt'


class Event:
    '''
    a simple class to manager the tremor's event catalog
    '''
    def __init__(self, chn, file=CATALOG):
        self.CATALOG = np.loadtxt(file)
        self.event = []
        self.chn = chn
        self.__filter()


    def __filter(self):

        self.event = []
        stat = Station()
        for i in range(self.CATALOG.shape[0]):
            log = self.CATALOG[i, :]
            # log[0~5] is time
            # log[6]=lon, log[7]=lat, log[8]=dep, log[9]=level, log[10]=flag

            # is earthquake event?
            if log[10] != 1:
                continue
            # not in area range and level is lower than 1, give up
            if ((log[6] > 134.5) or (log[6] < 131.8)) and ((log[7] > 34.5) or (log[7] < 32.5)):
                if log[9] < 0.5 or log[9] > 5:
                    continue
                else:
                    stat_near = stat.get_stations([log[6], log[7]], 40*3.78**(log[9]-1), self.chn)

            else:
                if log[9] > 3:
                    continue
                else:
                    stat_near = stat.get_stations([log[6], log[7]], 40*3.78**(log[9]-1), self.chn)

            if stat_near is not None:
                # all yes
                date = '%02d%02d%02d' % (log[0]-2000, log[1], log[2])
                hour = '%02d' % log[3]
                sec = log[4]*60+log[5]
                self.event.append({ 'date': date,
                                    'hour': hour,
                                    'sec': sec,
                                    'loc': [log[6], log[7], log[8]],
                                    'level': log[9],
                                    'station': stat_near})
            else:
                continue

            if len(self.event) >= 4000:
                break
            else:
                print(len(self.event))

# a = Event(chn=4)
# print(len(a.event))