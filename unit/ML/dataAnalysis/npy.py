## a npy data class based on numpy
import numpy as np

class npy():
    def __init__(self,path=None,shape = None,expand=16):
        self.shape = None
        self.data = None
        self.filePath = None
        self.expand=None
        
        if path!=None:
            self.loadnpy(path)
        if shape != None:
            self.reshape(shape,expand)
        
    def loadnpy(self, filePath):
        self.data = np.load(filePath)
        self.shape = self.data.shape
        self.filePath=filePath
        
    def reshape(self, shape,expand=16):
        shape = np.array(shape)
        
        data = np.zeros(shape+2*expand)
        
        data[expand:(shape[0]+expand),
             expand:(shape[1]+expand),
             expand:(shape[2]+expand)] = self.data.reshape([shape[0],shape[1],shape[2]])
             
        self.data = data
        self.shape = shape
        self.expand = expand
        
    
    def batch(self, batchNum, batchShape):
        pass
    
    def pickLayer(self,layerNum,sampleAxis = 13,NormRange=[-5e3,5e3],IMAGE=True):
        [XL,YL,TL] = self.shape
        if sampleAxis==13:
            a = self.data[self.expand:(XL+self.expand),
                          self.expand+layerNum,
                          self.expand:(TL+self.expand)]
        elif sampleAxis==23:
            a = self.data[self.expand+layerNum,
                           self.expand:(YL+self.expand),
                           self.expand:(TL+self.expand)]
        
        if IMAGE:
            record = a
            max = NormRange[1]
            min = NormRange[0]
            
            record = (record-min)/(max-min)*255.0
            record = np.round(record)
            
            maxIndex = np.where(record>255)
            minIndex = np.where(record<0)
            
            record[maxIndex] = 255.0;
            record[minIndex] = 0.0
            a = record
            
        return a

    def pick(self,index,shape,sampleAxis=13,NormRange=[-5e3,5e3],IMAGE=True):
        index = np.array(index)
        index=index+self.expand
        
        if   sampleAxis==13:
            record_sample = self.data[(index[0]-shape[0]//2):(index[0]+shape[0]-shape[0]//2),
                                      index[1],
                                      (index[2]-shape[1]//2):(index[2]+shape[1]-shape[1]//2)]
        elif sampleAxis==23:    
            record_sample =  self.data[index[0],
                                      (index[1]-shape[0]//2):(index[1]+shape[0]-shape[0]//2),
                                      (index[2]-shape[1]//2):(index[2]+shape[1]-shape[1]//2)]
        
        if IMAGE:
            record = record_sample
            max = NormRange[1]
            min = NormRange[0]
            
            record = (record-min)/(max-min)*255.0
            record = np.round(record)
            
            maxIndex = np.where(record>255)
            minIndex = np.where(record<0)
            
            record[maxIndex] = 255.0;
            record[minIndex] = 0.0
            record_sample = record
        
        return record_sample
        