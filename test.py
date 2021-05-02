import torch
import torch.nn as nn
from model import MyModel
from dataset import MyDataset
import os
import numpy as np
from torch.utils.data import DataLoader 
import pandas as pd 
hidden_size = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

my_model = MyModel(hidden_size).to(device)
criterion = nn.MSELoss()
my_model.load_state_dict(torch.load("./result/model.pth"))

def test():
    y_list = []
    loss_list =[]
    with open("predict.txt","w") as f:
        my_model.eval()
        my_dataset = MyDataset(csv_file = "data.csv")
        test_dataloader = DataLoader(dataset = my_dataset, batch_size= 1, shuffle= False, num_workers= 0)
        for idx , (x, label) in enumerate(test_dataloader):
            with torch.no_grad():
                label, x = label.to(device), x.to(device)
                output = my_model(x) 
                y = output.cpu().data.numpy().tolist()
                y_list.append(y)
                loss = criterion(output,label)
                loss_list.append(loss.item())
                f.write( '{}| output:{:.3f} | label: {:.3f} |loss: {:.5f} '.format( idx,output.item(), label.item(), loss.item()))
                f.write("\n")
    print("loss_avg:", np.mean(loss_list))
    y_numpy = np.array(y_list).reshape(-1,1)

    mid_np = np.array(y_numpy)                    #列表转数组
    mid_np_3f = np.round(mid_np,3)                 #对数组中的元素保留两位小数
    y_numpy = list(mid_np_3f)                     #数组转列表

    df = pd.DataFrame(y_numpy,columns=['Enhance'], index=None)
    df.to_csv("data_predict.csv")
    

test()