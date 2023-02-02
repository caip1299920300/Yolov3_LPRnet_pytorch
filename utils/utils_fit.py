import torch
from tqdm import tqdm
from utils.utils import get_lr

def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda):
    '''
        model_train:    整体模型
        model      :    backbone模型
        yolo_loss  :    损失函数
        loss_history:   日志函数
        optimizer   :   优化器        
        gen         :   训练集数据
        gen_val     :   验证集数据
        epoch_step  :   遍历一遍训练数据集需要的次数
        epoch_step_val: 遍历一遍验证数据集需要的次数
        epoch       :   起始epoch
        Epoch       :   最终epoch  
    '''
    # ---------------------------------------------------------------------------------------------------- #
    #                                              训练阶段                                                #
    # ---------------------------------------------------------------------------------------------------- #
    loss        = 0
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar: 
        for index, batch in enumerate(gen):
            if index >= epoch_step:
                break
            # 获取数据
            images, targets = batch[0], batch[1]
            # 对数据进行格式转换
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
        
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs         = model_train(images)
            # 计算损失
            loss_value_all  = 0 # 总损失值
            num_pos_all     = 0 # 得到损失正样本的个数
            for l in range(len(outputs)):
                #l->  0：13,13,512    1：26,26,256    2：52,52,128
                loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
                num_pos_all     += num_pos
            loss_value  = loss_value_all / num_pos_all
            # 反向传播
            loss_value.backward()
            optimizer.step()
            
            # 统计损失
            loss += loss_value.item()
            # tqtm输出信息
            pbar.set_postfix(**{'loss'  : loss / (index + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)
    print('Finish Train')


    # ---------------------------------------------------------------------------------------------------- #
    #                                              验证阶段                                                #
    # ---------------------------------------------------------------------------------------------------- #
    val_loss    = 0
    model_train.eval()  # 设置为验证模式
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for index, batch in enumerate(gen_val):
            if index >= epoch_step_val:
                break
            # 验证集数据
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            
                # 前向传播
                outputs         = model_train(images)
                # 计算损失
                loss_value_all  = 0
                num_pos_all     = 0
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                loss_value  = loss_value_all / num_pos_all
            val_loss += loss_value.item()
            # tqtm输出信息
            pbar.set_postfix(**{'val_loss': val_loss / (index + 1)})
            pbar.update(1)
    print('Finish Validation')
    
    
    # ---------------------------------------------------------------------------------------------------- #
    #                                           相关信息保存                                                #
    # ---------------------------------------------------------------------------------------------------- #
    # loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
