import matplotlib.pyplot as pyplot
import matplotlib.cm as cm 
import tensorflow as tf
import numpy as np
import nibabel as nb
import sys
import scipy.interpolate as interp
sys.path.append('/home/shi/FaultDetection/fault-test')

 
from dataAnalysis.nii import *
from dataAnalysis.npy import *
from dataAnalysis.segy import *
 
#model1 = nii('/home/shi/FaultDetection/data/SYN/FCN1_OA3_f10_n10/Nifti',expand=0,modelIndex=6)
 
#model2 = nii('/home/shi/FaultDetection/data/SYN/FCN2_OA3_f40_n10/Nifti',expand=0,modelIndex=6)
 

test = npy('/home/shi/FaultDetection/data/fault-test.npy')

'''
a = model1.record[10,10,:]
a = a/a.max()
b = model2.record[10,10,:]
b = b/b.max()

c = test.data[10,10,:]
cInterp = interp.interp1d(range(len(c)), c,kind="cubic")
c = cInterp(np.linspace(0,len(c)-1, 700))
c = c/c.max()

f1 = pyplot.figure()
pyplot.plot(range(len(a)),a, range(len(b)),b+5,range(len(c)),c+10)
pyplot.show()

aFFT = np.fft.fft(a)
bFFT = np.fft.fft(b)
cFFT = np.fft.fft(c)

f2 = pyplot.figure()
xa = np.linspace(0,np.pi*2, len(a))
xb = np.linspace(0,np.pi*2, len(b))
xc = np.linspace(0,np.pi*2, len(c))
pyplot.plot(xa,aFFT, xb,bFFT+100,xc,cFFT+200)
pyplot.show()


'''
t = [500,700,800,900,1000]
for inter in t:
    XL,YL,TL = test.shape
    data = np.zeros([XL,YL,inter])
    for i in range(XL):
        for j in range(YL):
            expand = 0
            k = test.data[i+expand,j+expand, expand:(TL+expand)]
            kInterp = interp.interp1d(range(len(k)), k,kind = 'cubic')
            
            k = kInterp(np.linspace(0,len(k)-1, inter))
            data[i,j,:] = k
        print(inter,i)
    
    np.save('/home/shi/FaultDetection/data/fault-test-interp%g'%inter,data)
    del data
