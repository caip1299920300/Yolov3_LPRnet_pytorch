import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor

class YoloDataset(Dataset):
    def __init__(self, annotations_lines, input_shape, num_classes, train):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotations_lines
        self.input_shape      = input_shape
        self.num_classes      = num_classes
        self.length           = len(self.annotation_lines)
        self.train            = train
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        index       = index % self.length   # 防止数据溢出
        
        #----------------------图像增强---------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image, box  = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        image       = np.transpose(np.array(image, dtype=np.float32)/255.0, (2, 0, 1)) 
        box         = np.array(box, dtype=np.float32) 
        if len(box) != 0:
            # 坐标缩放到0-1
            box[:, [0,2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1,3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]   # 宽高
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4]/2 # 中心坐标
            
        return image, box
    
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        line    = annotation_line.split()
        # 读取图像并转换为RGB图像
        image   = Image.open(line[0])
        image   = cvtColor(image)
        
        # 获取图像的宽高和目标宽高
        iw, ih  = image.size
        h, w    = input_shape
        
        # 获取预测框
        box     = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])


        if not random:  # 如果不是训练数据的话，则直接给图片加灰条，修改图片大小
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)
            dx      = (w-nw) // 2
            dy      = (h-nh) // 2
            # 将图片多余的部分加上灰条
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx,dy))
            image_data  = np.array(new_image, np.float32)
            # 对真实框进行调整
            if len(box) > 0:
                np.random.shuffle(box)  # 随机打乱真实框
                box[:, [0,2]] = box[:, [0,2]]*nw/iw  + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih  + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w]     = w
                box[:, 3][box[:, 3]>h]     = h
                box_w   = box[:, 2] - box[:, 0]
                box_h   = box[:, 3] - box[:, 1]
                box     = box[np.logical_and(box_w>1, box_h>1)] # 筛选掉比较小边框
            return image_data, box
    
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        #   将图像多余的部分加上灰条
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image
        # 坐标变化
        if len(box)>0:
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # 坐标变化
            if len(box)>0:
                box[:, [0,2]] = w - box[:, [2,0]]	# w为image的宽
        
        #------------------------------------------#
        #   色域扭曲(不改变坐标)
        #------------------------------------------#
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        
        #---------------------------------#
        #   最后对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)	# 随机打乱真实框
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w	# 如果超出边界，则直接去边界值
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]	# 去除很小的目标
        
        return image_data, box

            
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes