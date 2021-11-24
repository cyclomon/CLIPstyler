import torch
import torch.nn as nn
import torchvision
import numpy as np

class UpBlock(nn.Module):
    def __init__(self, in_channel=128, out_channel=64):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Sequential(          
                                    nn.Conv2d(out_channel+in_channel, out_channel,
                                                  3, 1,1),
            nn.InstanceNorm2d(out_channel),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channel, out_channel,
                                                  3, 1,1),
            nn.InstanceNorm2d(out_channel),
                                      nn.ReLU()
                                      )
        
        self.skip = nn.Conv2d(out_channel+in_channel,out_channel,1,1,0)

    def forward(self, input,FB_in):
        out_temp = self.upsample(input)
        out_temp = torch.cat([out_temp,FB_in],dim=1)
        out = self.conv(out_temp) + self.skip(out_temp)
        
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channel=3, out_channel=64):
        super().__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel,3, 1,1),
                                  nn.InstanceNorm2d(out_channel),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channel, out_channel,3, 1,1),
                                  nn.InstanceNorm2d(out_channel),
                                      nn.ReLU()
                                      )
        self.skip = nn.Conv2d(in_channel,out_channel,1,1,0)
        # self.downsample = nn.MaxPool2d(2,2)
        self.downsample = nn.Conv2d(out_channel,out_channel,4,2,1)

    def forward(self, input):
        out_temp = self.conv(input) + self.skip(input)
        out = self.downsample(out_temp)
        return out,out_temp

class EncodingBlock(nn.Module):
    def __init__(self, in_channel=256, out_channel=512):
        super().__init__()
        
        self.conv = nn.Sequential(
                                    nn.Conv2d(in_channel, out_channel,
                                                  3, 1,1),
                                      # nn.BatchNorm2d(out_channel),
                                    nn.InstanceNorm2d(out_channel),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channel, out_channel,
                                                  3, 1,1),
            nn.InstanceNorm2d(out_channel),
                                      nn.ReLU()
                                      )
        self.skip = nn.Conv2d(in_channel,out_channel,1,1,0)
    def forward(self, input):
        out = self.conv(input) + self.skip(input)
        return out
    
class UNet(nn.Module):
    def __init__(self, ngf=16, input_channel=3, output_channel=3):
        super(UNet, self).__init__()
        self.conv_init = nn.Conv2d(input_channel,ngf,1,1,0)
        self.init = EncodingBlock(ngf,ngf)
        self.down1 = DownBlock(ngf,ngf)
        self.down2 = DownBlock(ngf,2*ngf)
        self.down3 = DownBlock(2*ngf,4*ngf)
        
        self.encoding = EncodingBlock(4*ngf,8*ngf)
        self.up3 = UpBlock(8*ngf,4*ngf)
        self.up2 = UpBlock(4*ngf,2*ngf)
        self.up1 = UpBlock(2*ngf,ngf)
        self.out = EncodingBlock(2*ngf,ngf)
        self.conv_fin = nn.Conv2d(ngf,output_channel,1,1,0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,use_sigmoid=True):
        x = self.conv_init(x)
        x = self.init(x)
        d1,d1_f = self.down1(x)
        d2,d2_f = self.down2(d1)
        d3,d3_f = self.down3(d2)
        
        h = self.encoding(d3)
        hu3 = self.up3(h,d3_f)
        hu2 = self.up2(hu3,d2_f)
        hu1 = self.up1(hu2,d1_f)
        
        h = self.out(torch.cat([hu1,x],dim=1))
        h = self.conv_fin(h)      
        if use_sigmoid:
            h = self.sigmoid(h)
        return h