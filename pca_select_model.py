# -*- coding: utf-8 -*-
"""
DAE model and PCA selection layers
RenMin 20190918
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shortcut=True):
        super(EncoderBlock, self).__init__()
        self.shortcut = shortcut
        if shortcut:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                          #nn.BatchNorm2d(out_channels)
                                          )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1 ,bias=False)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.nolinear1 = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1 ,bias=False)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.nolinear2 = nn.PReLU(out_channels)

        
    def forward(self, x):
        if self.shortcut:
            shortcut_x = self.shortcut_layer(x)
        #x = self.nolinear1(self.bn1(self.conv1(x)))
        #x = self.nolinear2(self.bn2(self.conv2(x)))
        x = self.nolinear1(self.conv1(x))
        x = self.nolinear2(self.conv2(x))
        if self.shortcut:
            x = x + shortcut_x
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.in_layer = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1, bias=False),
                                      nn.PReLU(64))
        
        self.block1 = EncoderBlock(64, 64, stride=2)
        self.block2 = EncoderBlock(64, 64, stride=1)
        self.block3 = EncoderBlock(64, 128, stride=2)
        self.block4 = EncoderBlock(128, 128, stride=1)
        self.block5 = EncoderBlock(128, 256, stride=2)
        self.block6 = EncoderBlock(256, 256, stride=1)
        self.block7 = EncoderBlock(256, 512, stride=2)
        self.block8 = EncoderBlock(512, 512, stride=1)
        
        #self.out_layer = nn.Sequential(Flatten(),
                                    #nn.Linear(256*7*7, 1024),
                                    #nn.BatchNorm1d(1024),
                                    #nn.PReLU(1024))
        
    def forward(self, x):
        #pdb.set_trace()
        x = self.in_layer(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        #x = self.out_layer(x)
        return x



class PCASelection(nn.Module):
    def __init__(self):
        super(PCASelection, self).__init__()
        self.fc1 = nn.Sequential(Flatten(),
                                nn.Linear(512*7*7, 512),
                                #nn.BatchNorm1d(512),
                                nn.PReLU(512))
        self.fc2 = nn.Sequential(nn.Linear(512, 2500),
                                #nn.BatchNorm1d(2500),
                                nn.Sigmoid())
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output=False):
        super(DecoderBlock, self).__init__()        
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.nolinear1 = nn.PReLU(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1 ,bias=False)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        if output:
            self.nolinear2 = nn.Tanh()
        else:
            self.nolinear2 = nn.PReLU(out_channels)

        
    def forward(self, x):
        #pdb.set_trace()
        #x = self.nolinear1(self.bn1(self.conv1(x)))
        #x = self.nolinear2(self.bn2(self.conv2(x)))
        x = self.nolinear1(self.conv1(x))
        x = self.nolinear2(self.conv2(x))
        return x

class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()
        self.encoder1 = nn.Sequential(EncoderBlock(1, 64, stride=2, shortcut=False),
                                     EncoderBlock(64, 128, stride=2, shortcut=False),
                                     EncoderBlock(128, 256, stride=2, shortcut=False))
        self.decoder1 = nn.Sequential(DecoderBlock(256, 128),
                                     DecoderBlock(128, 64),
                                     DecoderBlock(64, 1, output=True))
        self.encoder2 = nn.Sequential(EncoderBlock(1, 64, stride=2, shortcut=False),
                                     EncoderBlock(64, 128, stride=2, shortcut=False),
                                     EncoderBlock(128, 256, stride=2, shortcut=False))
        self.decoder2 = nn.Sequential(DecoderBlock(256, 128),
                                     DecoderBlock(128, 64),
                                     DecoderBlock(64, 1, output=True))
        
    def forward(self, x):
        #pdb.set_trace()
        x1 = self.encoder1(x)
        x1 = self.decoder1(x1)
        x2 = self.encoder1(x1)
        x2 = self.decoder1(x2)

        return x1, x2


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
