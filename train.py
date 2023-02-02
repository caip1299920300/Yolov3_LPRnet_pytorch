import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


from nets.yolonet import YoloBody
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes, weights_init
from utils.utils_fit import fit_one_epoch
from nets.loss import YOLOLoss


# ------------------------------------------------------------------------------------- #
#                                 1、相关的配置信息                                      #
# ------------------------------------------------------------------------------------- #
# 是否启用Cuda加速训练
Cuda            = True
#   设置使用多线程读取数据的个数
num_workers         = 0
# 预训练模型地址
model_weigths_path    = 'model_data/ep010-loss1.900-val_loss1.586.pth'
# 数据集的种类配置文件地址
classes_path    = 'model_data/my_classes.txt'
# anchor配置文件地址
anchors_path    = 'model_data/yolo_anchors.txt'
#   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# 网络的输入大小
input_shape     = [416, 416]        
# 骨干网络是否加载预训练权重
pretrained      = True
#   图片路径和标签地址
train_annotation_path   = 'CCPD2019/tran.txt'
val_annotation_path     = 'CCPD2019/val.txt'


# 冻结式训练
Freeze_Train    = True
# 数据训练批次加载的数据个数
batch_size      = 8
# 设置学习率
lr_Nobackbone   = 1e-3
lr_all          = 1e-4

# ------------------------------------------------------------------------------------- #
#                                 2、读取相关信息                                        #
# ------------------------------------------------------------------------------------- #
#   读取classes和anchor
class_names, num_classes = get_classes(classes_path)
anchors, num_anchors     = get_anchors(anchors_path)

# ------------------------------------------------------------------------------------- #
#                                 3、加载数据集                                          #
# ------------------------------------------------------------------------------------- #
#   读取数据集对应的txt
with open(train_annotation_path) as f:
    train_lines = f.readlines()
with open(val_annotation_path) as f:
    val_lines   = f.readlines()
num_train   = len(train_lines)
num_val     = len(val_lines)
print("*"*50)
print(num_train,num_val)
print("*"*50)

epoch_step      = num_train // batch_size
epoch_step_val  = num_val // batch_size
if epoch_step == 0 or epoch_step_val == 0:
    raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

train_dataset   = YoloDataset(train_lines, input_shape, num_classes, train = True)
val_dataset     = YoloDataset(val_lines, input_shape, num_classes, train = False)
gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                            drop_last=True, collate_fn=yolo_dataset_collate)
gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                            drop_last=True, collate_fn=yolo_dataset_collate)


# ------------------------------------------------------------------------------------- #
#                                 4、初始化损失函数                                      #
# ------------------------------------------------------------------------------------- #
yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]])


# ------------------------------------------------------------------------------------- #
#                                 5、初始化网络                                          #
# ------------------------------------------------------------------------------------- #
model = YoloBody(anchors_mask,num_classes,pretrained=pretrained)
if model_weigths_path != '':    # 加载预训练模型
    print('Load weights {}.'.format(model_weigths_path))
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(model_weigths_path, map_location = device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
elif not pretrained:  # 骨干网络不加载预训练权重，则直接初始化模型参数
        weights_init(model)

model_train = model.train() # 设置为训练模式
if Cuda:
    model_train = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model_train = model_train.cuda()
    
# ---------------------------------------------------------------------------------------------------- #
#                                 6、优化器设置                                                         #
# ---------------------------------------------------------------------------------------------------- #
if Freeze_Train:
    # 非主干网络的优化器
    optimizer_Nobackbone       = optim.Adam(model_train.parameters(), lr_Nobackbone, weight_decay = 5e-4)
    lr_scheduler_Nobackbone    = optim.lr_scheduler.StepLR(optimizer_Nobackbone, step_size=1, gamma=0.94)

# 整个网络的优化器
optimizer_all       = optim.Adam(model_train.parameters(), lr_all, weight_decay = 5e-4)
lr_scheduler_all    = optim.lr_scheduler.StepLR(optimizer_all, step_size=1, gamma=0.94)

# ----------------------------------------------------------------------------------------------------- #
#                                         7、训练                                                        #
# ----------------------------------------------------------------------------------------------------- #
# 冻结backbone主干网络
start_epoch     = 0
end_epoch       = 50
if Freeze_Train:
    for param in model.backbone.parameters():
        param.requires_grad = False
    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model_train, model, yolo_loss, None, optimizer_Nobackbone, epoch, 
                epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
        lr_scheduler_Nobackbone.step()

# 训练全部网络
start_epoch     = 50
end_epoch       = 100
for param in model.backbone.parameters():
    param.requires_grad = True
for epoch in range(start_epoch, end_epoch):
    fit_one_epoch(model_train, model, yolo_loss, None, optimizer_all, epoch, 
            epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
    lr_scheduler_all.step()