import numpy as np
from analysis.station import Station

CATALOG = r'/media/pkushi/86847781-60d2-4f71-8ed9-24e2b48d79e1/data/jma1cat_flag.txt'


class Tremor:
    '''
    a simple class to manager the tremor's event catalog
    '''
    def __init__(self, chn, file=CATALOG, area='sg'):
        self.CATALOG = np.loadtxt(file)
        self.tremor = []
        self.chn = chn
        self.area = area
        self.__filter()

    def __filter(self):

        self.tremor = []
        stat = Station(AREA=self.area)
        for i in range(self.CATALOG.shape[0]):
            log = self.CATALOG[i, :]
            # log[0~5] is time
            # log[6]=lon, log[7]=lat, log[8]=dep, log[9]=level, log[10]=flag

            # is tremor?
            if log[10] != 5:
                continue
            # in area range?

            if log[10] != 5:
                continue
            # in area range?
            if not self.is_in_area(log):
                continue
            # have enough stations?
            stat_near = stat.get_stations([log[6], log[7]], 50, self.chn)

            if stat_near is not None:
                # all yes
                date = '%02d%02d%02d' % (log[0]-2000, log[1], log[2])
                hour = '%02d' % log[3]
                sec = log[4]*60+log[5]
                self.tremor.append({'date': date,
                                    'hour': hour,
                                    'sec': sec,
                                    'loc': [log[6], log[7], log[8]],
                                    'station': stat_near})

    def is_in_area(self, log):

        result = True
        if self.area == 'sg':
            if (log[6] > 134.5) or (log[6] < 131.8):
                result = False
            if (log[7] > 34.5) or (log[7] < 32.5):
                result = False

        if self.area == 'kii':
            if (log[6] > 136.8) or (log[6] < 134.8):
                result = False
            if (log[7] > 35) or (log[7] < 33.4):
                result = False

        if self.area == 'archi':
            if (log[6] > 138.4) or (log[6] < 136.8):
                result = False
            if (log[7] > 35.6) or (log[7] < 34.6):
                result = False
        return result

# a = tremor()
# print(a.tremor)