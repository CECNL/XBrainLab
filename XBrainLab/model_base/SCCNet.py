import torch
import torch.nn as nn
import math
import numpy as np

class SCCNet(nn.Module):
    """Implementation of SCCNet
    https://ieeexplore.ieee.org/document/8716937

    Parameters:
        n_classes: Number of classes.
        channels: Number of channels.
        samples: Number of samples.
        sfreq: Sampling frequency.
        Ns: Number of spatial filters.
    """
    def __init__(self, n_classes, channels, samples, sfreq, Ns=22):
        super(SCCNet, self).__init__() # input:bs, 1, channel, sample

        self.tp = samples
        self.ch = channels
        self.sf = sfreq
        self.n_class = n_classes
        self.octsf = int(math.floor(self.sf*0.1))


        # (1, n_ch, kernelsize=(n_ch,1))
        self.conv1 = nn.Conv2d(1, Ns, (self.ch, 1))  
        self.Bn1 = nn.BatchNorm2d(Ns) #(n_ch) 
        #kernelsize=(1, floor(sf*0.1)) padding= (0, floor(sf*0.1)/2)
        self.conv2 = nn.Conv2d(Ns, 20, (1, self.octsf), padding=(0, int(np.ceil((self.octsf-1)/2))))
        self.Bn2   = nn.BatchNorm2d(20)
        
        self.Drop1 = nn.Dropout(0.5) 
        #kernelsize=(1, sf/2) revise to 128/2?  stride=(1, floor(sf*0.1))
        self.AvgPool1 = nn.AvgPool2d((1, int(self.sf/2)), stride=(1, int(self.octsf))) 
        # (20* ceiling((timepoint-sf/2)/floor(sf*0.1)), n_class)
        self.classifier = nn.Linear(20* int( (self.tp + (int(np.ceil((self.octsf-1)/2)) * 2 - self.octsf + 1) - int(self.sf/2) ) / int(self.octsf)+1 ), self.n_class, bias=True)
        
        self.addHook = False
        self.spatialGrad = None
        self.spatialResponse = None
        self.temporalGrad = None
        self.temporalResponse = None
        
        
    def setHook(self, addHook):
        self.addHook = addHook
    
    def forward(self, x):
        def convert(var):
            return var.detach().cpu().numpy().squeeze()
        def spGrad(var):
            self.spatialGrad = convert(var)
        def tpGrad(var):
            self.temporalGrad = convert(var)
            
        if len(x.shape) != 4:
            x = x.unsqueeze(1)
        spX = self.conv1(x) #(128,22,1,562)
        # hook for sptial grad
        if self.addHook:
            spX.register_hook(spGrad)
            self.spatialResponse = convert(spX)
            
        x = self.Bn1(spX)
        tpX = self.conv2(x) #(128,20,1,563)
        # hook for temperal grad
        if self.addHook:
            tpX.register_hook(tpGrad)
            self.temporalResponse = convert(tpX)
        
        x = self.Bn2(tpX)
        x = x ** 2
        x = self.Drop1(x)
        x = self.AvgPool1(x) #(128,20,1,42)
        x = torch.log(x) 
        x = x.view(-1, 20* int( (self.tp + (int(np.ceil((self.octsf-1)/2)) * 2 - self.octsf + 1) - int(self.sf/2) ) / int(self.octsf)+1 ))
        x = self.classifier(x)

        return x
    
    def getTemporalWeights(self):
        weights = self.conv2.weight.detach().cpu().numpy().squeeze()
        return weights

    def getSpatialWeights(self):
        weights = self.conv1.weight.detach().cpu().numpy().squeeze()
        return weights