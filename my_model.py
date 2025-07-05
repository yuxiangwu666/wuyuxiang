import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import read_feature_label as rfl
class conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,3,1,1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out,ch_out,3,1,1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv(x)
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,3,1,1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
    def forward(self,x):
        return self.up(x)
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,F_int,1,1,0),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,F_int,1,1,0),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi
class Encoder(nn.Module):
    def __init__(self, ch_in=3, ch_out=256):  # 修改最终输出通道数为 256
        super(Encoder, self).__init__()
        self.conv1 = conv(ch_in, 32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv(32, 64)
        self.conv3 = conv(64, 128)
        self.conv4 = conv(128, ch_out)  # 移除原来的 conv5
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        x4 = self.pool(x3)
        x4 = self.conv4(x4)  # 最后一层为 conv4
        return x4, x3, x2, x1  # 返回的最高层为 x4

class Decoder(nn.Module):
    def __init__(self, ch_in=256, ch_out=1):  # 修改输入通道数为 256
        super(Decoder, self).__init__()
        self.up4 = up_conv(ch_in, 128)  # 修改起始上采样层
        self.att4 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.up_conv4 = conv(256, 128)
        self.up3 = up_conv(128, 64)
        self.att3 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.up_conv3 = conv(128, 64)
        self.up2 = up_conv(64, 32)
        self.att2 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.up_conv2 = conv(64, 32)
        self.conv1 = nn.Conv2d(32, ch_out, 3, 1, 1)
    def forward(self, x):
        x4, x3, x2, x1 = x  # 修改输入为 x4 开始
        d4 = self.up4(x4)
        x3 = self.att4(d4, x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)
        d3 = self.up3(d4)
        x2 = self.att3(d3, x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)
        d2 = self.up2(d3)
        x1 = self.att2(d2, x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        d1 = self.conv1(d2)
        return d1

class attention_unet(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super(attention_unet, self).__init__()
        self.encoder = Encoder(ch_in=in_channels)
        self.decoder = Decoder(ch_out=out_channels)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x