from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from obspy.geodetics.base import calc_vincenty_inverse as circle_distance

STATS_INF_FILE = '/media/pkushi/86847781-60d2-4f71-8ed9-24e2b48d79e1/data/NIED_SeismicStation_20180702.csv'

class Station:
    def __init__(self, station_file=None, AREA='sg'):
        # all valid statons, 87
        self.valid_station_index = {'sg':['N.AIOH', 'N.AYKH', 'N.BSEH', 'N.DWAH', 'N.GEIH', 'N.GHKH', 'N.GSIH', 'N.HHIH', 'N.HIKH', 'N.HISH',
                                          'N.HIYH', 'N.HKBH', 'N.HNSH', 'N.HRSH', 'N.HSMH', 'N.HWSH', 'N.IKKH', 'N.IKNH', 'N.IKTH', 'N.IKWH',
                                          'N.INOH', 'N.IWAH', 'N.JNSH', 'N.KHUH', 'N.KKGH', 'N.KMGH', 'N.KNBH', 'N.KNGH', 'N.KNNH', 'N.KNSH',
                                          'N.KRHH', 'N.KSAH', 'N.KTGH', 'N.KTWH', 'N.KURH', 'N.KWBH', 'N.KYDH', 'N.MABH', 'N.MHRH', 'N.MIGH',
                                          'N.MIHH', 'N.MISH', 'N.MITH', 'N.MKIH', 'N.MNOH', 'N.MTYH', 'N.MURH', 'N.MYKH', 'N.NAKH', 'N.NMKH',
                                          'N.OHCH', 'N.OKMH', 'N.OKYH', 'N.OOTH', 'N.OOZH', 'N.SADH', 'N.SETH', 'N.SGUH', 'N.SINH', 'N.SJOH',
                                          'N.SKIH', 'N.SSKH', 'N.SYTH', 'N.TAMH', 'N.TBEH', 'N.TBRH', 'N.TGUH', 'N.TKBH', 'N.TOHH', 'N.TOKH',
                                          'N.TSMH', 'N.TSSH', 'N.TSYH', 'N.TTAH', 'N.UUMH', 'N.UWAH', 'N.YNDH', 'N.YSHH'],
                                    'aichi':['N.TYEH', 'N.ASHH', 'N.HOUH', 'N.NUKH', 'N.ASUH', 'N.TDEH', 'N.OKZH', 'N.HRYH', 'N.STRH', 'N.NGKH',
                                             'N.KSHH', 'N.ANJH'],
                                    'kii':['N.HNZH', 'N.KTDH', 'N.OTOH', 'N.KAWH', 'N.URSH', 'N.HYSH', 'N.INMH', 'N.HRKH', 'N.MGWH', 'N.GNOH',
                                           'N.OWSH', 'N.MASH', 'N.OYMH', 'N.KRTH', 'N.TKEH', 'N.TKWH']
                                    }
        self.valid_station = self.valid_station_index[AREA]
        self.inf = {}
        if station_file is None:
            self.get_stat_location()
        else:
            self.get_stat_location(station_file)

    def get_stat_location(self, filename=STATS_INF_FILE):
        # read all stations' information from a STATS_INF_FILE
        f = open(filename, errors='ignore')
        inf = f.readlines()
        f.close()
        inf.pop(0)

        stat = {}
        for i in inf:
            inf_i = i.split(',')
            name = inf_i[2]
            latitude = float(inf_i[7])
            longitude = float(inf_i[8])
            h = float(inf_i[11])
            if stat.get(name) is None:
                stat[name] = [longitude, latitude, h]
            else:
                raise ValueError('station name repeated: %s \n %s' % (name, i))

        for i in self.valid_station:
            if self.inf.get(i) is not None:
                raise ValueError('station name repeated or you are rewrite: %s' %i)
            else:
                self.inf[i] = stat[i]

    def get_stations(self, event_loc, radius, num):
        # get the stations arround a event(loc: [lon,lat]), radius is KM
        if num > len(self.valid_station):
            raise ValueError('num is bigger than valid stations\' number')

        stat_with_dis = []
        for i in self.valid_station:
            stat_inf_i = self.inf[i]
            dis_i, _, _ = circle_distance(lon1=stat_inf_i[0], lon2=event_loc[0], lat1=stat_inf_i[1], lat2=event_loc[1])
            stat_with_dis.append([i, dis_i/1000])

        # sort from small(index=0) to big(index=-1)
        stat_with_dis.sort(key=lambda x: x[1], reverse=False)

        stat_return = []
        for i in range(num):
            if stat_with_dis[i][1] <= radius:
                stat_return.append(stat_with_dis[i])
            else:
                # don't have num stations in the circle
                return None
        return stat_return


# filename_sample = 'C:\\Users\\shiyx\\Documents\\Data\\tremor\\data\\NIED_SeismicStation_20180702.csv'
#
# map = Basemap(llcrnrlon=125, llcrnrlat=25, urcrnrlon=150, urcrnrlat=50)
# map.arcgisimage(service='Ocean_Basemap',xpixels=100000, verbose=True)
# #map.bluemarble()
# stat_lon = []
# stat_lat = []
# stat_name = []
# for i in stat.keys():
#     stat_name.append(i)
#     stat_lon.append(stat[i][0])
#     stat_lat.append(stat[i][1])
# x, y = map(stat_lon, stat_lat)
#
# for i in range(len(stat_name)):
#     map.scatter(x, y, 4, marker='.', color='k')
# plt.savefig('C:\\Users\\shiyx\\Documents\\Data\\tremor\\data\\stations.png')
# plt.show()



