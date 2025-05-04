from torch.utils.data import Dataset,DataLoader
import numpy as np
import h5py
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import torch
from PIL import Image
import torchvision.transforms as transforms
def get_labels(is_train=True):
    # 将数据的sha256号码和标号做成一个元组,将ben数据和mal数据进行shuffle并根据最小的哪一个进行限制保证数据量1:1,之后将他们变成列表并返回.
    if is_train:
        file_path = r"E:\malware\malex\labels_zipped\MaleXv2-m1789632-b306244_02052018.hdf5"
        ret = "训练集"
    else:
        file_path = r"E:\malware\malex\labels_zipped\MaleXv1-m864669-b179725_10062017.hdf5"
        ret = "验证集"
    with h5py.File(file_path, 'r') as file:
        list_ben = [(i.decode('utf-8'),0) for i in file['ben_sha256s']]
        list_mal = [(i.decode('utf-8'),1) for i in file['mal_sha256s']]
    length_ben = len(list_ben)
    length_mal = len(list_mal)
    restrict = min(length_ben,length_mal)
    print(f'这里是{ret},ben_sha256的长度为:{length_ben},mal_sha256的长度为:{length_mal},总长度为:{length_mal+length_ben}')
    # random.shuffle(list_ben)
    # random.shuffle(list_mal)
    ret_tul_li = list_ben[:restrict]+list_mal[:restrict]
    random.shuffle(ret_tul_li)
    return ret_tul_li

def transform2Tensor():
    return transforms.Compose([  
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.5], std=[0.5]),  # 归一化
        transforms.ConvertImageDtype(torch.float32)
    ])

class MalexDataset(Dataset):
    def __init__(self,is_train=True,transforms=None):
        self.name_label_tuple = get_labels(is_train)[:50000]
        self.transform = transforms
        self.length = len(self.name_label_tuple)
        self.dir_bigram =r'E:\malware\malex\data\bigram_data'
        self.dir_byteplot = r'E:\malware\malex\data\byteplot_data'
    def __len__(self):
        return self.length
    def __getitem__(self,index):
        name,label = self.name_label_tuple[index]
        path_bigram =os.path.join(self.dir_bigram,name+'_bigram_dct.png')
        path_byteplot = os.path.join(self.dir_byteplot,name+'_img_256.png')
        if os.path.exists(path_bigram) and os.path.exists(path_byteplot):
            bigram = self.transform(Image.open(path_bigram))
            byteplot = self.transform(Image.open(path_byteplot))
        else:
            bigram = torch.from_numpy(np.zeros((256,256),dtype = np.float32)).unsqueeze(0)
            byteplot = torch.from_numpy(np.zeros((256,256),dtype = np.float32)).unsqueeze(0)
            label = 0
        return bigram,byteplot,torch.tensor(label,dtype=torch.float32)
        

if __name__ == "__main__":
    dataset = MalexDataset(is_train=False,transforms=transform2Tensor())
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True,num_workers=4)
    for bigram,byteplot,label in tqdm(dataloader):
        print(label)
        