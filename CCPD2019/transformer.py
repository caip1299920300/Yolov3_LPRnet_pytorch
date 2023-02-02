# coding:utf-8
import os
import os.path
import re
import shutil

import cv2
from tqdm import tqdm


def listPathAllfiles(dirname):
    with open(dirname,'r') as file:
        result = file.readlines()
    result = [i.strip() for i in result]
    return result

def trans_label(data_path,class_num):
    images_files = listPathAllfiles(data_path)
    labels_txt=""
    for name in tqdm(images_files):
        if name.endswith(".jpg") or name.endswith(".png"):
            img = cv2.imread(name.strip())
            height, width = img.shape[0], img.shape[1]

            str1 = re.findall('-\d+\&\d+_\d+\&\d+-', name)[0][1:-1]
            str2 = re.split('\&|_', str1)
            x0 = int(str2[0])
            y0 = int(str2[1])
            x1 = int(str2[2])
            y1 = int(str2[3])
            # 转为中心坐标，宽高模式
            # x = round((x0 + x1) / 2 / width, 6)
            # y = round((y0 + y1) / 2 / height, 6)
            # w = round((x1 - x0) / width, 6)
            # h = round((y1 - y0) / height, 6)

            # 转为yolov3的数据格式
            labels_txt = labels_txt+name+" %s,%s,%s,%s,%s"%(x0,y0,x1,y1,str(class_num))+"\n"
    return labels_txt
        

if __name__ == '__main__':
    # 训练集
    data_path = r'splits\train.txt'  # 数据集在哪里需要填写
    labels_txt = trans_label(data_path,1)
    with open("./tran.txt", "w") as file:
        file.write(labels_txt)
    # 验证集
    data_path = r'splits\val.txt'  # 数据集在哪里需要填写
    labels_txt = trans_label(data_path,1)
    with open("./val.txt", "w") as file:
        file.write(labels_txt)
    # 测试集
    data_path = r'splits\test.txt'  # 数据集在哪里需要填写
    labels_txt = trans_label(data_path,1)
    with open("./test.txt", "w") as file:
        file.write(labels_txt)
