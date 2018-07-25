import numpy as np
import matplotlib.pyplot as plt

from analysis.getdata import Wave
from analysis.tremor import Tremor
from analysis.earthquake import Event
a = Tremor(chn=4)
b = Wave()

index = 102
a_tremor = a.tremor[index]
data = []
a_tremor['station'].sort(key=lambda x: x[1], reverse=True)
print(a_tremor)

duration = 3600
shift = -duration/2
for i in a_tremor['station']:
    sac = b.get_waveform(a_tremor, duration=duration, shift=shift, station=i[0])[0]
    sac = sac-sac.mean()
    wave_data = sac*1e7

    plt.plot(np.linspace(0, duration, len(wave_data)), wave_data+i[1], label=i[0])

plt.axvline(x=-shift)

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)



plt.figure()
event = Event(4)
print(len(event.event))
index = 102
a_event = event.event[index]
data = []
a_event['station'].sort(key=lambda x: x[1], reverse=True)
print(a_event)

duration = 3600
shift = -duration/2
for i in a_event['station']:
    sac = b.get_waveform(a_event, duration=duration, shift=shift, station=i[0])[0]
    sac = sac-sac.mean()
    wave_data = sac/sac.max()

    plt.plot(np.linspace(0, duration, len(wave_data)), wave_data+i[1], label=i[0])

plt.axvline(x=-shift)

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)


plt.show()