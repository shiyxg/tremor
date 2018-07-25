import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.patches as patches
from analysis.station import Station
from analysis.tremor import Tremor
file_path = r'C:\Users\shiyx\Documents\Data\tremor\data'
path_char = '\\'

mpl.rcParams['font.size'] = 10.
mpl.rcParams['font.family'] = 'Comic Sans MS'
mpl.rcParams['axes.labelsize'] = 8.
mpl.rcParams['xtick.labelsize'] = 6.
mpl.rcParams['ytick.labelsize'] = 6.

fig = plt.figure(figsize=(14, 6), dpi=200)
ax = fig.add_subplot(111)
map = Basemap(llcrnrlon=130, llcrnrlat=32, urcrnrlon=140, urcrnrlat=36, resolution='h', ax=ax)
map.readshapefile(file_path+path_char+r'gadm36_JPN_shp\gadm36_JPN_0', 'japan', linewidth=0.2)
map.drawparallels(np.linspace(32,36, 5),labels=[1, 0, 1, 0], color='black', dashes=[1,0], linewidth=0.)
map.drawmeridians(np.linspace(130, 140, 11),labels=[1, 0, 1, 0], color='black', dashes=[1,0], linewidth=0.)

stat = Station(station_file=file_path+'\\NIED_SeismicStation_20180702.csv')
tremor = Tremor(4, file=r'C:\Users\shiyx\Documents\Data\tremor\data\jma1cat_flag.txt')

# sample = tremor.tremor[3]
# ex, ey = map(sample['loc'][0], sample['loc'][1])
# map.scatter(ex, ey, s=4, marker='*', color='r', label='tremor')
# region = patches.Rectangle([131.8, 32.5], 2.7, 2, fill=None, color='tab:gray', label='tremor region')
# circle = patches.Circle(xy=sample['loc'], radius=0.5, fill=None, color='r', label='station region')
# plt.gca().add_patch(region)
# plt.gca().add_patch(circle)
# ex,ey = stat.inf['N.IKWH'][0], stat.inf['N.IKWH'][1]
# region = patches.Rectangle([131.8, 32.5], 2.7, 2, fill=None, color='tab:gray', label='tremor region')
# circle = patches.Circle(xy=(ex, ey), radius=0.5, fill=None, color='r', label='group region')
# plt.gca().add_patch(region)
# plt.gca().add_patch(circle)

region1 = patches.Rectangle([131.8, 32.5], 2.7, 2, fill=None, color='r', label='SK')
region2 = patches.Rectangle([134.8, 33.4], 2, 1.6, fill=None, color='g', label='KII')
region3 = patches.Rectangle([136.8, 34.6], 1.8, 1, fill=None, color='b', label='AICHI')
plt.gca().add_patch(region1)
plt.gca().add_patch(region2)
plt.gca().add_patch(region3)
l = True
# for i in stat.valid_station:
#     stat_lon = stat.inf[i][0]
#     stat_lat = stat.inf[i][1]
#     x, y = map(stat_lon, stat_lat)
#     # for j in sample['station']:
#     #     if i == j[0]:
#     #         if l:
#     #             map.scatter(x, y, s=4, marker='^', color='b', label='picked stations')
#     #             l = False
#     #         else:
#     #             map.scatter(x, y, s=4, marker='^', color='b')
#     #
#     #         plt.annotate(i, xy=(x, y), xycoords='data', xytext=(x, y), fontsize=4, color='r')
#     #         break
#     #     else:
#     #         map.scatter(x, y, s=4, marker='^', color='tab:gray')
#
#     # group = ['N.IKWH', 'N.SADH', 'N.SINH', 'N.AYKH']
#     #
#     # if i in group:
#     #     map.scatter(x, y, s=4, marker='^', color='r')
#     # elif (stat_lon > 134.5) or (stat_lon < 131.8) or (stat_lat > 34.5) or (stat_lon < 32.5):
#     #     map.scatter(x, y, s=4, marker='^', color='tab:gray')
#     # else:
#     #     map.scatter(x, y, s=4, marker='^', color='b')
#
# # map.scatter(ex, ey, s=4, marker='*', color='r', label='group center')
# # map.scatter(ex, ey, s=4, marker='^', color='r', label='group stations')
# # map.scatter(0, 0, s=4, marker='^', color='b', label='potential stations')
# # map.scatter(0, 0, s=4, marker='^', color='tab:gray', label='unpicked stations')
for i in ['N.AIOH', 'N.AYKH', 'N.BSEH', 'N.DWAH', 'N.GEIH', 'N.GHKH', 'N.GSIH', 'N.HHIH', 'N.HIKH', 'N.HISH',
                              'N.HIYH', 'N.HKBH', 'N.HNSH', 'N.HRSH', 'N.HSMH', 'N.HWSH', 'N.IKKH', 'N.IKNH', 'N.IKTH', 'N.IKWH',
                              'N.INOH', 'N.IWAH', 'N.JNSH', 'N.KHUH', 'N.KKGH', 'N.KMGH', 'N.KNBH', 'N.KNGH', 'N.KNNH', 'N.KNSH',
                              'N.KRHH', 'N.KSAH', 'N.KTGH', 'N.KTWH', 'N.KURH', 'N.KWBH', 'N.KYDH', 'N.MABH', 'N.MHRH', 'N.MIGH',
                              'N.MIHH', 'N.MISH', 'N.MITH', 'N.MKIH', 'N.MNOH', 'N.MTYH', 'N.MURH', 'N.MYKH', 'N.NAKH', 'N.NMKH',
                              'N.OHCH', 'N.OKMH', 'N.OKYH', 'N.OOTH', 'N.OOZH', 'N.SADH', 'N.SETH', 'N.SGUH', 'N.SINH', 'N.SJOH',
                              'N.SKIH', 'N.SSKH', 'N.SYTH', 'N.TAMH', 'N.TBEH', 'N.TBRH', 'N.TGUH', 'N.TKBH', 'N.TOHH', 'N.TOKH',
                              'N.TSMH', 'N.TSSH', 'N.TSYH', 'N.TTAH', 'N.UUMH', 'N.UWAH', 'N.YNDH', 'N.YSHH']:
    stat_lon = stat.inf[i][0]
    stat_lat = stat.inf[i][1]
    x, y = map(stat_lon, stat_lat)
    map.scatter(x, y, s=4, marker='^', color='r')
    plt.annotate(i, xy=(x, y), xycoords='data', xytext=(x, y), fontsize=4, color='r')
map.scatter(0, 0, s=4, marker='^', color='r', label='sk stations')

for i in ['N.TYEH', 'N.ASHH', 'N.HOUH', 'N.NUKH', 'N.ASUH', 'N.TDEH', 'N.OKZH', 'N.HRYH', 'N.STRH',
                              'N.NGKH','N.KSHH', 'N.ANJH',]:
    stat_lon = stat.inf[i][0]
    stat_lat = stat.inf[i][1]
    x, y = map(stat_lon, stat_lat)
    map.scatter(x, y, s=4, marker='^', color='b')
    plt.annotate(i, xy=(x, y), xycoords='data', xytext=(x, y), fontsize=4, color='b')
map.scatter(0, 0, s=4, marker='^', color='b', label='aichi stations')

for i in ['N.HNZH', 'N.KTDH', 'N.OTOH', 'N.KAWH', 'N.URSH', 'N.HYSH', 'N.INMH', 'N.HRKH', 'N.MGWH',
                              'N.GNOH','N.OWSH', 'N.MASH', 'N.OYMH', 'N.KRTH', 'N.TKEH', 'N.TKWH']:
    stat_lon = stat.inf[i][0]
    stat_lat = stat.inf[i][1]
    x, y = map(stat_lon, stat_lat)
    map.scatter(x, y, s=4, marker='^', color='g')
    plt.annotate(i, xy=(x, y), xycoords='data', xytext=(x, y), fontsize=4, color='g')
map.scatter(0, 0, s=4, marker='^', color='g', label='kii stations')

plt.title('stations and tremor regions of three areas', y = 1.08)
plt.legend()
plt.savefig(file_path+'\\TTT_stations.png')
plt.show()
