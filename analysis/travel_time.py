import os
import numpy as np

PATH = 'C:\\Users\\shiyx\\Documents\\Data\\tremor\\travelTime\\'
EXE_NAME = 'get_travel_time.exe'


class TravelTime:
    def __init__(self, path=PATH, program_name=EXE_NAME):
        self.exe = program_name
        self.path = path

    def get_travel_time(self, delta, source_depth):

        for i in delta:
            command = '.\\'+self.exe + ' %s'%i + ' %s'%source_depth
            os.system(command)
        result = np.loadtxt('output.txt')

        os.system('DEL output.txt')
        return result[:, -1]


# a = TravelTime()
# b = a.get_travel_time(range(1000), 10)
# print(b)