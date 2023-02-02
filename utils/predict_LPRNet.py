# -*- coding: utf-8 -*-
from LPRNet.LPRNet import build_lprnet
from torch.autograd import Variable
from torch.utils.data import *
import numpy as np
import torch
import cv2

class PLPRnet():
    def __init__(self,pretrained_model):
        self.CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
                    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
                    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
                    '新',
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                    'W', 'X', 'Y', 'Z', 'I', 'O', '-'
                    ]
        # 加载模型
        self.lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=len(self.CHARS), dropout_rate=0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lprnet.to(device)
        print("Successful to build LPRNet!")

        # load pretrained model
        if pretrained_model:
            self.lprnet.load_state_dict(torch.load(pretrained_model))
            print("load pretrained LPRNet successful!")
        else:
            print("[Error] LPRNet Can't found pretrained mode, please check!")
            return False
        
    def transform(self,img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)  # 转成tensor类型
        img = img.unsqueeze(0)  # 添加batch维度 1*24*94*3
        return img
    
    def predict(self,Image_):
        img_size = (94, 24)
        # 推理
        Image_ = cv2.cvtColor(np.asarray(Image_), cv2.COLOR_RGB2BGR)
        height, width, _ = Image_.shape
        if height != img_size[1] or width != img_size[0]:
            Image_ = cv2.resize(Image_, img_size)
        images = self.transform(Image_)
        if torch.cuda.is_available():
            images = Variable(images.cuda())
        else:
            images = Variable(images)
        # forward
        with torch.no_grad():
            prebs = self.lprnet(images)
            # greedy decode
            prebs = prebs.cpu().detach().numpy()
            preb = prebs[0, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(self.CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(self.CHARS) - 1):
                    if c == len(self.CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
        # show image and its predict label
        lb = ""
        for i in no_repeat_blank_label:
            lb += self.CHARS[i]
        # print(lb)
        return lb


