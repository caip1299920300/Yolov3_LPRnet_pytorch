import os
import random
import xml.etree.ElementTree as ET
from utils.utils import get_classes

VOCdevkit_sets  = [('2012', 'train'), ('2012', 'val')]

# 数据集地址
VOCdevkit_path = r'F:\5.AI_project\datas\VOCdevkit'

# 读取数据集总类
classes_path        = 'model_data/voc_classes.txt'
classes,_ = get_classes(classes_path)

# 训练测试比例
trainval_percent    = 0.9   # 训练集：测试集
train_percent       = 0.9   # 训练集训练部分：训练集验证部分

def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    
    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
    
if __name__ == '__main__':
    random.seed(0)
    
    print("Generate txt in ImageSets.")
    xmlfilepath     = os.path.join(VOCdevkit_path, 'VOC2012/Annotations')
    saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2012/ImageSets/Main')
    
    temp_xml        = os.listdir(xmlfilepath)
    total_xml       = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)
    
    num     = len(total_xml)
    list    = range(num)
    tv      = int(num * trainval_percent)
    tr      = int(tv * train_percent)   
    trainval= random.sample(list, tv)
    train   = random.sample(trainval, tr)
    
    print("train and val size",tv)
    print("train size",tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
    
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")

    
    print("Generate 2012_train.txt and 2012_val.txt for train.")
    for year, image_set in VOCdevkit_sets:
        image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
        list_file = open('model_data/%s_%s.txt' %(year, image_set), 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))
            # 解读xml标签文件
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
        list_file.close()
    print("Generate 2012_train.txt and 2012_val.txt for train done.")
