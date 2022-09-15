import torch
import torch.nn as nn
import math

class ShallowConvNet(nn.Module):
    def __init__(self, n_classes, channels, samples, sfreq):
        super(ShallowConvNet, self).__init__()
        
        self.temporal_filter = 40
        self.spatial_filter = 40
        self.kernel = math.ceil(sfreq * 0.1)
        
        self.conv1 = nn.Conv2d(1, self.temporal_filter, (1, self.kernel), bias=False)
        self.conv2 = nn.Conv2d(self.temporal_filter, self.spatial_filter, (channels, 1), bias=False)
        self.Bn1   = nn.BatchNorm2d(self.spatial_filter)
        # self.SquareLayer = square_layer()
        self.AvgPool1 = nn.AvgPool2d((1, 35), stride=(1, 7))
        # self.LogLayer = Log_layer()
        self.Drop1 = nn.Dropout(0.25)
        self.outSize = (samples - self.kernel) - 35 / 7
        self.classifier = nn.Linear(self.spatial_filter*self.outSize, n_classes, bias=True)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Bn1(x)
        x = x ** 2
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = self.Drop1(x)
        x = x.view(-1, self.spatial_filter*self.outSize)
        x = self.classifier(x)

        #x = self.softmax(x)
        return x