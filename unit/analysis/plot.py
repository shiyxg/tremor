# A toolkit to compute the great-circle distance
# https://docs.obspy.org/packages/autogen/obspy.geodetics.base.calc_vincenty_inverse.html

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from analysis.station import Station
event = '2017040119 35.0289, 136.9710, 1153.827'
lat_e = 35.0289
lon_e = 136.9710
time = '1704011900'
file_path = r'C:\Users\shiyx\Documents\Data\tremor\data'
data_path = r'C:\Users\shiyx\Documents\Data\tremor\17040119'
path_char = '\\'

fig = plt.figure(figsize=(10, 8), dpi=100)
ax = fig.add_subplot(111)
map = Basemap(llcrnrlon=130, llcrnrlat=32, urcrnrlon=136, urcrnrlat=36, resolution=None, ax=ax)
map.readshapefile('/home/pkushi/gm-jpn-bnd_u_2/coastl_jpn', 'japan')
map.drawparallels(np.linspace(32,36, 5),labels=[1,0,1,0])
map.drawmeridians(np.linspace(130, 136, 7),labels=[1,0,1,0])
# map.drawlsmask()
stat = Station()
stat_lon = []
stat_lat = []
stat_name = []
for i in stat.valid_station:
    stat_name.append(i)
    stat_lon.append(stat.inf[i][0])
    stat_lat.append(stat.inf[i][1])
x, y = map(stat_lon, stat_lat)

for i in range(len(stat_name)):
    map.scatter(x, y, 30, marker='^', color='k', label=stat_name[i])
# plt.savefig('C:\\Users\\shiyx\\Documents\\Data\\tremor\\data\\stations.png')
# plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.show()
# stats_near = []
# for i in stats_keys:
#     lat_s = stats[i][1]
#     lon_s = stats[i][0]
#     dis_i, _, _ = distance(lat1=lat_e, lon1= lon_e, lat2=lat_s, lon2=lon_s)
#     dis_i = dis_i/1000
#     print(i)
#     print(dis_i)
#     if dis_i < 250:
#         stats_near.append([i, dis_i])
#
# data_near = []
# for i in stats_near:
#     file = data_path+path_char+time+'_'+i[0]+'_U.s'
#     stat_data = ob.read(file)[0].data
#     data_near.append([i[0],i[1], stat_data])
#
# data_near.sort(key=lambda x: x[1], reverse=True)
# for i in data_near:
#     i[2] = i[2]/i[2].max()
#     line = plt.plot(np.linspace(0, 1, len(stat_data)), i[2]*2+i[1], label=i[0], linewidth=0.5)
# plt.xlabel('time(h)')
# plt.ylabel('delta(KM)')
# plt.title(time)
# plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
# plt.show()



