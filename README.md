# Yolov3_LPRnet_pytorch
**(如果你使用git下载总是报红，你可以选择下载dev-sidecar编程软件加速下载--https://gitee.com/interesting-goods/dev-sidecar?_from=gitee_search)**

**国内gitee仓库地址：** https://gitee.com/wucaip/Yolov3_LPRnet_pytorch

## 一、基于yolov3的LPRnet车牌识别

以yolov3作文检测器、LPRnet作为识别模型，实现了实时车牌车牌识别模型。本项目直接使用LPRnet原始的网络作为车牌识别模型。
## 二、数据集介绍
本项目使用的是CCPD2019数据集中的ccpd_base常用车牌数据集作为训练集，所以只能识别蓝底的中国车牌。

## 三、模型预测

> 1.下载模型（下载解压，放在model_data文件夹中）
>
> 链接：https://pan.baidu.com/s/1sARkEgcWt4tpatAqGLXWEA 提取码：6eu6
>
> 2.CMD进入命令行模式，运行 predict.py  预测脚本

~~~
python  predict.py
~~~

> 输入预测图片的地址:  img/test.jpg

<table rules="none" align="center"> 	
    <tr> 		
        <td> 			
            <center> 				
                <img src=".\img\test.jpg" width="100%" /> 				
                <br/> 				
                <font color="AAAAAA">原始图片</font> 			
            </center> 		
        </td> 		
        <td> 			
            <center> 				
                <img src=".\img\test_predict1.jpg" width="100%" /> 				
                <br/> 				
                <font color="AAAAAA">识别后的图片</font> 			
            </center> 		
        </td> 	
    </tr> 
</table>


## 四、检测模型训练

> 1.下载数据集，并解压后，将CCPD2019文件夹整个文件复制到本目录的CCPD2019中
>
> 链接：https://pan.baidu.com/s/1QdNG-iqIhZzSWdlOvS9gxQ 提取码：ymd3

> 2.解压标签
>
> 解压CCPD2019中的lable.zip脚本

> 3.下载模型（下载解压，放在model_data文件夹中）
>
> 链接：https://pan.baidu.com/s/1sARkEgcWt4tpatAqGLXWEA 提取码：6eu6
>
> 4.运行训练脚本

~~~
python train.py
~~~


## 五、目录介绍

Project:

│  predict.py   # 预测脚本

│  README.md    

│  requirements.txt # 本项目所需要的库

│  train.py         # 训练脚本

│  voc_annotation.py    # 对Voc数据集预测里脚本

│

├─img

│      test.jpg

│

├─model_data

│      ep010-loss1.900-val_loss1.586.pth    # yolo网络模型

│      Final_LPRNet_model.pth  # LPRNet网络模型

│      simhei.ttf       # 预测画框的字体文件

│      my_classes.txt  # 数据集种类文本

│      yolo_anchors.txt # anchor文件

├─CCPD2019

│	  transformer.py  #CCPD2019生成标签的脚本

│

├─nets

│  │  darknet.py        # yolo骨干网络

│  │  loss.py           # yolo的损失函数脚本

│  │  yolonet.py        # yolo整体网络

├─LPRNet

│  │  LPRNet.py        # 车牌识别骨干网络

│

└─utils

    │  dataloader.py    # 加载数据集以及预处理脚本
    
    │  predict_yolo.py  # yolo模型预测函数脚本
    
    │  predict_LPRNet.py# 车牌识别模型预测函数脚本
    
    │  utils.py         # 真个项目需要用到的函数脚本
    
    │  utils_bbox.py    # 模型预测的预测框解码以及nms函数脚本
    
    │  utils_fit.py     # 训练函数脚本


# 六、待改进

- [ ] 使用CCPD2019全部数据集+CCPD2020数据集进行训练
- [ ] 使用CCPD2019+CCPD2020数据集重新训练LPRnet网络

# 七、参考

1. https://github.com/sirius-ai/LPRNet_Pytorch
2. https://github.com/bubbliiiing/yolo3-pytorch
