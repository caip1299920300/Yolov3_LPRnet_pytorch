import colorsys
import torch
import torch.nn as nn
import numpy as np
from PIL import ImageDraw, ImageFont
from utils.utils import (get_classes, get_anchors, cvtColor, resize_image)
from utils.utils_bbox import DecodeBox
from nets.yolonet import YoloBody
from utils.predict_LPRNet import PLPRnet

class YOLO(object):
    _defaults = {
        # 模型权重地址
        # "model_path"        : r'model_data/new_yolo_weigth.pth',
        "model_path"        : r'model_data/ep010-loss1.900-val_loss1.586.pth',
        "LPRNet_model_path" : r'model_data/Final_LPRNet_model.pth',
        # 种类地址
        "classes_path"      : 'model_data/my_classes.txt',
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #   输入图片的大小，必须为32的倍数。
        "input_shape"       : [416, 416],
        #   只有得分大于置信度的预测框会被保留下来
        "confidence"        : 0.5,
        #   非极大抑制所用到的nms_iou大小
        "nms_iou"           : 0.3,
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        "letterbox_image"   : True,
        #   是否使用Cuda
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        # 用于设置属性值，该属性不一定是存在的
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        #   获得种类和先验框的数量
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #   画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   建立车牌识别模型
        #---------------------------------------------------#
        self.PLPRnet = PLPRnet(self.LPRNet_model_path)
        #---------------------------------------------------#
        #   建立yolov3模型，载入yolov3模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        
    
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        # 将图像转换成RGB图像
        image       = cvtColor(image)   
        # 给图像增加灰条，实现不失真的resize
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #   添加上batch_size维度并归一化
        image_data  = np.expand_dims(np.transpose(np.array(image_data, dtype='float32')/255. , (2, 0, 1)), 0)
    
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #   将图像输入网络当中进行预测
            outputs = self.net(images)
            #   将获得的信息解码成真实的大小
            outputs = self.bbox_util.decode_box(outputs)
            #   将预测框进行堆叠，然后进行非极大抑制
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
            
        #   设置字体与边框厚度
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        
        
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]
            
            top, left, bottom, right = box
            
            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            # label = '{} {:.2f}'.format(predicted_class, score)
            img_LPR = image.crop((max(0, np.floor(left).astype('int32')-2),max(0, np.floor(top).astype('int32')-2),min(image.size[0], np.floor(right).astype('int32')+2),min(image.size[1], np.floor(bottom).astype('int32')+2)))
            label = self.PLPRnet.predict(img_LPR)
            # img_LPR.show()
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            # 画框调整
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
                
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image
    
