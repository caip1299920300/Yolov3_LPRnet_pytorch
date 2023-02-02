import torch
import torch.nn as nn
from nets.darknet import darknet53

def Conv_bn_relu(filters_in, filters_out, kernel_size):
    pad = (kernel_size -1) // 2 if kernel_size else 0
    return nn.Sequential(*[
        nn.Conv2d(filters_in, filters_out, kernel_size=kernel_size,
                  stride=1,padding=pad,bias=False),
        nn.BatchNorm2d(filters_out), 
        nn.LeakyReLU(0.1)
    ])


#------------------------------------------------------------------------#
#   make_last_layers里面五个卷积用于提取特征。
#   两个卷积用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def YoloHead(filters_list, in_filters, out_filter):
    Conv_5 = nn.Sequential(*[
        Conv_bn_relu(in_filters, filters_list[0], 1),
        Conv_bn_relu(filters_list[0], filters_list[1], 3),
        Conv_bn_relu(filters_list[1], filters_list[0], 1),
        Conv_bn_relu(filters_list[0], filters_list[1], 3),
        Conv_bn_relu(filters_list[1], filters_list[0], 1)
    ])
    conv_2 = nn.Sequential(*[
        Conv_bn_relu(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1,
                  padding=0, bias=True)
    ])
    return Conv_5,conv_2


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(YoloBody, self).__init__()
         #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53()
        if pretrained:  # 主干网络加载预训练模型
            self.backbone.load_state_dict(torch.load("model_data/darknet53.pth"))
            
        #-------------------------------------------------------------------------#
        #   获取Darknet网络的输出层通道out_filters : [64, 128, 256, 512, 1024]
        #-------------------------------------------------------------------------#
        out_filters = self.backbone.layers_out_filters
        
        
        self.last_layer0,self.last_head0 = YoloHead([512, 1024], out_filters[-1], len(anchors_mask[0])*(num_classes+5))
        
        self.last_layer1_conv            = Conv_bn_relu(512,256,1)
        self.last_layer1_upsample        = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1,self.last_head1 = YoloHead([256, 512], out_filters[-2]+256, len(anchors_mask[1])*(num_classes+5))
        
        self.last_layer2_conv            = Conv_bn_relu(256,128,1)
        self.last_layer2_upsample        = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2,self.last_head2 = YoloHead([128, 256], out_filters[-3]+128, len(anchors_mask[2])*(num_classes+5))
        
        
    def forward(self, x):
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   x2:52,52,256；x1:26,26,512；x0:13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)
        

        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0_branch = self.last_layer0(x0)
        out0        = self.last_head0(out0_branch)
        
        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)
        
        
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1(x1_in)
        out1        = self.last_head1(out1_branch)
        
        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        

        #---------------------------------------------------#
        #   第三个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2 = self.last_layer2(x2_in)
        out2 = self.last_head2(out2)
        
        return out0,out1,out2
    

# if __name__ == '__main__':
#     anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
#     model = YoloBody(anchors_mask,20)
#     datas = torch.rand((1,3,416,416))
#     print(datas.shape)
#     result = model(datas)
#     print([i.shape for i in result])

