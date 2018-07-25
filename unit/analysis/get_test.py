import numpy as np
import os

test_num = 1000
noise_num = 7296
events_num = 12000
tremors_num = 5364

num = 14232
test_data = []
for i in range(test_num):
    name = '%s.npy'%(num-i-1)
    test_data.append(name)
    command = 'mv '+ '/home/pkushi/dataset_T_1d1s3c/sg/'+name + ' /home/pkushi/dataset_T_1d1s3c_test/sg/%s.npy'%i
    os.system(command)
    print(i)