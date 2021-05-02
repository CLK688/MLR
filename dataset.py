#创建数据集
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd 
import torchvision.transforms as transforms

class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data_nor = self.data.iloc[:,1:].values
        _, n = self.data_nor.shape #获取特征数
        """standardize normalization"""
        for i in range(n):    
            features = self.data_nor[:,i]
            self.meanVal = features.mean(axis = 0)
            self.std = features.std(axis = 0)
            if self.std != 0:
                self.data_nor[:,i] = (features - self.meanVal) / self.std
            else :
                self.data_nor[:,i] = 0
        """Min-Max normalization"""
        # for i in range(n):
        #     features = self.data_nor[:,i]
        #     minVal = features.min(axis = 0)
        #     maxVal = features.max(axis = 0)
        #     diff = maxVal - minVal
        #     if diff != 0:
        #         self.data_nor[:,i] = (features - minVal) / diff
        #     else :
        #         self.data_nor[:,i] = 0

    def __getitem__(self, index):
        label = self.data.iloc[index, -1]
        label = label.astype('float32').reshape(-1)
        data = self.data_nor[index, 0:-1]
        data = data.astype('float32').reshape(-1) 
        
        return data, label
    def __len__(self):
        return len(self.data)

    def get_nor(self):
        return self.meanVal, self.std

# my_dataset = MyDataset("data.csv")
# train_dataloader = DataLoader(dataset = my_dataset, batch_size=4, shuffle= False, num_workers= 0,)
# for i, (data, label) in enumerate(train_dataloader):

#     print(data,label,data.size())
#     break
    


