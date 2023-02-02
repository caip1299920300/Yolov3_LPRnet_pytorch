import torch.nn as nn

#---------------------------------------------------------------------#
#   卷积+BN+LeakyRule
#---------------------------------------------------------------------#
class Convolutional(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=3, stride=1, padding=0):
        super(Convolutional, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False)
        self.bn     = nn.BatchNorm2d(out_channels)
        self.lk_relu= nn.LeakyReLU(0.1)
    
    def forward(self, x):
        out = self.conv2d(x)
        out = self.bn(out)
        out = self.lk_relu(out)
        
        return out

#---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
#---------------------------------------------------------------------#
class Residual(nn.Module):
    def __init__(self,in_c, out_cs):
        super(Residual, self).__init__()
        self.Conv_bn_rule1 = Convolutional(in_c, out_cs[0],kernel_size=1)
        self.Conv_bn_rule2 = Convolutional(out_cs[0], out_cs[1],kernel_size=3,padding=1)
    
    def forward(self, x):
        residual = x
        
        out      = self.Conv_bn_rule1(x)
        out      = self.Conv_bn_rule2(out)
        
        out += residual
        return out
        

class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        # 416,416,3 -> 416,416,32
        self.Conv_bn_rule = Convolutional(3, 32,kernel_size=3,padding=1)
        # 416,416,32 -> 208,208,64
        self.layer1 = self._blocks_layers([32, 64], layers[0])
        # 208,208,64 -> 104,104,128
        self.layer2 = self._blocks_layers([64, 128], layers[1])
        # 104,104,128 -> 52,52,256
        self.layer3 = self._blocks_layers([128, 256], layers[2])
        # 52,52,256 -> 26,26,512
        self.layer4 = self._blocks_layers([256, 512], layers[3])
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._blocks_layers([512, 1024], layers[4])
        
        # 各个block输出的通道数
        self.layers_out_filters = [64, 128, 256, 512, 1024]
    
    def _blocks_layers(self,channels, blocks):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(Convolutional(channels[0], channels[1],kernel_size=3,stride=2,padding=1))
        # 加入残差结构
        for i in range(0, blocks):
            layers.append(Residual(channels[1],channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.Conv_bn_rule(x)
        out1 = self.layer1(x)       # 208,208,64
        out2 = self.layer2(out1)    # 104,104,128
        out3 = self.layer3(out2)    # 52,52,256
        out4 = self.layer4(out3)    # 26,26,512
        out5 = self.layer5(out4)    # 13,13,1024
        
        return out3,out4,out5
    
def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model

if __name__ == '__main__':
    import torch
    model =darknet53()
    model.load_state_dict(torch.load("model_data/darknet53.pth"))
    datas = torch.rand((1,3,416,416))
    print(datas.shape)
    result = model(datas)
    print([i.shape for i in result])

