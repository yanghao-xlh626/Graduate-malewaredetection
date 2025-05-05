import os
from tqdm import tqdm
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader,random_split
from PIL import Image
from matplotlib.animation import FuncAnimation
import torch
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append(r'E:\malware\malex\tools')
from Animate import Animate
from Data import MalexDataset,transform2Tensor,get_labels
from Moud import MalexResnet,MalexBigram,MalexByteplot,ResidualBlock
from Monitor import EarlyStopping
import os
from tqdm import tqdm

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def device_init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device.type)
    return device

def save_checkpoint(model, optimizer, animator, path,epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'animator_state_dict': animator.save_state_dict(),  # 保存自定义状态
        'epoch':epoch
    }, path)
    print(f"已保存检查点至路径:{path}")

def load_checkpoint(model, optimizer, animator,path,new_learn_rate,device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    animator.load_state_dict(checkpoint['animator_state_dict'])  # 恢复图形数据
    epoch = checkpoint['epoch']  # 恢复训练的起始周期
    for param_group in optimizer.param_groups:
        if 'lr' in param_group:
            param_group['lr'] = new_learn_rate
            write_log(f"学习率已更新为:{new_learn_rate}\n",graph_path)
        else:
            write_log("学习率参数未找到\n",graph_path)  # 记录日志信息,说明学习率参数未找到
    return epoch

def checkpoint_init(batch_size,max_epochs,learn_rate,graph_path):
    device = device_init()  # 确保在调用 device_init() 后使用 devic
    model = MalexResnet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    animator = Animate(batchsize=batch_size, learn_rate=learn_rate,max_epochs=max_epochs,graph_path =graph_path)
    criterion = nn.BCEWithLogitsLoss()
    epoch =0
    return model, optimizer, animator,device,criterion,epoch

def write_log(content,log_folder):
    with open(os.path.join(log_folder,'log.txt'),'a',encoding='utf-8') as f:
        f.write(content)
        print(content)
def train_model(dataset, batch_size, learn_rate, max_epochs, num_workers, save_path,load_path,graph_path):
    total_size = dataset.__len__()
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 初始化模型、优化器和损失函数等
    model,optimizer,animator,device,criterion,epoch =checkpoint_init(batch_size,max_epochs,learn_rate,graph_path)
    early_stopping = EarlyStopping(patience=5, delta=0.001)
    try:
        epoch = load_checkpoint(model, optimizer, animator, load_path,learn_rate)
        print("检查点文件已导入")
        write_log(f"检查点文件已导入,epoch为{epoch}\n",graph_path)
    except :  # 处理文件未找到的情况,或者没有epoch的情况
        print("首次训练,无检查点文件用以导入")
        write_log("首次训练,无检查点文件用以导入\n",graph_path)

    try:
        for current_epoch in tqdm(range(epoch+1,max_epochs+1),desc='Epoch'):
            model.train()
            train_loss = 0.0
            tqdm.write('训练阶段')
            for bigrams,byteplots,labels in tqdm(train_loader,desc='Batch',leave=False):
                bigrams = bigrams.to(device)  # 确保数据在 GPU 上
                byteplots = byteplots.to(device)  # 确保数据在 GPU 上
                labels = labels.view(-1).float().to(device)  # 确保数据在 GPU 上
                assert bigrams.device == next(model.parameters()).device and byteplots.device ==next(model.parameters()).device, f"数据设备 {bigrams.device},{byteplots.device} 和模型设备 {next(model.parameters()).device} 不一致"
                assert labels.device == next(model.parameters()).device, f"数据设备 {labels.device} 和模型设备 {next(model.parameters()).device} 不一致"
                assert isinstance(bigrams, torch.Tensor), f"输入数据类型错误: {type(bigrams)}"
                optimizer.zero_grad()
                outputs = model(bigrams,byteplots)  # 确保模型在 GPU 上
                loss = criterion(outputs,labels)
                # print('输出为:',outputs,'实际为',labels)  #BUG
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * bigrams.size(0)  # 计算总损失
            train_loss /= len(train_loader.dataset)  # 计算平均损失

            # 在测试集上评估模型
            model.eval()
            test_loss=0.0
            correct = 0
            tqdm.write('测试阶段')
            with torch.no_grad():
                for bigrams,byteplots,labels in tqdm(test_loader,desc='Batch',leave=False):
                    bigrams = bigrams.to(device)  # 确保数据在 GPU 上
                    byteplots = byteplots.to(device)  # 确保数据在 GPU 上
                    labels = labels.to(device)  # 确保数据在 GPU 上
                    outputs = model(bigrams,byteplots)  # 确保模型在 GPU 上
                    loss = criterion(outputs,labels)
                    test_loss += loss.item() * bigrams.size(0)  # 计算总损失
                    preds = (torch.sigmoid(outputs)>0.5).float()
                    # print(f'预测为{preds},实际为{labels}')          #BUG
                    correct += (preds==labels).sum().item()
            test_loss /= len(test_loader.dataset)  # 计算平均损失
            accuracy = correct / len(test_loader.dataset)  # 计算准确率

            # TODO:t添加了一个早停机制
            early_stopping(test_loss, model)
            if early_stopping.early_stop:
                print("触发早停")
                write_log("触发早停\n",graph_path)
                break
            
            animator.update(current_epoch,train_loss,test_loss,accuracy)
            tqdm.write(f'Epoch {current_epoch}/{max_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
            write_log(f'Epoch{current_epoch}/{max_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}\n',graph_path)
            save_checkpoint(model, optimizer, animator, save_path,current_epoch)  # 保存每个周期的模型
        animator.finalize()
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        write_log(f"训练过程中发生错误: {e}\n",graph_path)
        save_checkpoint(model, optimizer, animator, save_path,current_epoch)  # 保存当前状态,确保在发生错误时保存模型状态


if __name__ == "__main__":


    '''graph_path = r'E:\malware\malex\checkpoint\Resnet_V1'
    load_path = r'E:\malware\malex\checkpoint\Resnet_V1\checkpoint.pth'  # 加载模型的路径
    save_path = r'E:\malware\malex\checkpoint\Resnet_V1\checkpoint.pth'  # 保存模型的路
    #此处因为出现过严重过拟合,精度为0,训练损失很低(拟合效果极好),测试损失到处跳.V1
    train_model(dataset,batch_size,learn_rate,max_epochs,num_workers,save_path,load_path,graph_path)'''



    '''# 残差块添加了一个dropout为0.1查看效果.发现训练损失可以看见在慢慢下降,但是测试损失和精度没了
    graph_path = r'E:\malware\malex\checkpoint\Resnet_V2'
    load_path = r'E:\malware\malex\checkpoint\Resnet_V2\checkpoint.pth'  # 加载模型的路径
    save_path = r'E:\malware\malex\checkpoint\Resnet_V2\checkpoint.pth'  # 保存模型的路
    train_model(dataset,batch_size,learn_rate,max_epochs,num_workers,save_path,load_path,graph_path)'''

    '''# 尝试归纳模型添加了sigmoid,结果精度为1,三条线都是水平线
    graph_path = r'E:\malware\malex\checkpoint\Resnet_V3'
    load_path = r'E:\malware\malex\checkpoint\Resnet_V3\checkpoint.pth'  # 加载模型的路径
    save_path = r'E:\malware\malex\checkpoint\Resnet_V3\checkpoint.pth'  # 保存模型的路
    train_model(dataset,batch_size,learn_rate,max_epochs,num_workers,save_path,load_path,graph_path)'''

    '''# 修改了dataloader,让他进行shuffle.解决了数据只有0的问题,
    # 回退并删除了sigmoid,训练了18个epoch,精度为0.85左右,损失下降不明显,增大数据集至2万,
    # 发现模型,精度为0.8273,不收敛,增大数据集至2.5万,将学习率从0.001改为0.0001,训练30-46个epoch发现过拟合开始明显,
    # 将数据集增加到5万,从46训练到60看看.触发早停精度稳定在0.8653,开始过拟合
    dataset = MalexDataset(is_train = True,transforms=transform2Tensor())
    batch_size = 16
    learn_rate = 0.0001
    max_epochs = 60
    num_workers = 4
    graph_path = r'E:\malware\malex\checkpoint\Resnet_V4'
    load_path = r'E:\malware\malex\checkpoint\Resnet_V4\checkpoint.pth'  # 加载模型的路径
    save_path = r'E:\malware\malex\checkpoint\Resnet_V4\checkpoint.pth'  # 保存模型的路
    train_model(dataset,batch_size,learn_rate,max_epochs,num_workers,save_path,load_path,graph_path)'''

    '''# 将block的dropout从0.3调到0.5,并将block中的dropout放到第三个batchnorm后,精最高度到达0.8766.
    dataset = MalexDataset(is_train = True,transforms=transform2Tensor())
    batch_size = 16
    learn_rate = 0.001
    max_epochs = 60
    num_workers = 4
    graph_path = r'E:\malware\malex\checkpoint\Resnet_V5'
    load_path = r'E:\malware\malex\checkpoint\Resnet_V5\checkpoint.pth'  # 加载模型的路径
    save_path = r'E:\malware\malex\checkpoint\Resnet_V5\checkpoint.pth'  # 保存模型的路
    train_model(dataset,batch_size,learn_rate,max_epochs,num_workers,save_path,load_path,graph_path)'''


    '''# 尝试增加层数,增加了两个block,并修改resnet的fc层为5120*1024,最高精度0.8669,最终精度0.8663
    dataset = MalexDataset(is_train = True,transforms=transform2Tensor())
    batch_size = 16
    learn_rate = 0.001
    max_epochs = 60
    num_workers = 4
    graph_path = r'E:\malware\malex\checkpoint\Resnet_V6'
    load_path = r'E:\malware\malex\checkpoint\Resnet_V6\checkpoint.pth'  # 加载模型的路径
    save_path = r'E:\malware\malex\checkpoint\Resnet_V6\checkpoint.pth'  # 保存模型的路
    train_model(dataset,batch_size,learn_rate,max_epochs,num_workers,save_path,load_path,graph_path)'''


    # 两个branch分别减少两个block,在合并后怎加一个全连接层,让他更加平滑.第一轮训练11个epoch精度到0.8751,改用学习率0.00001训练
    dataset = MalexDataset(is_train = True,transforms=transform2Tensor())
    batch_size = 16
    learn_rate = 0.00001
    max_epochs = 60
    num_workers = 4
    graph_path = r'E:\malware\malex\checkpoint\Resnet_V7'
    load_path = r'E:\malware\malex\checkpoint\Resnet_V7\checkpoint.pth'  # 加载模型的路径
    save_path = r'E:\malware\malex\checkpoint\Resnet_V7\checkpoint.pth'  # 保存模型的路
    train_model(dataset,batch_size,learn_rate,max_epochs,num_workers,save_path,load_path,graph_path)