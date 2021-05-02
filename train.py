import torch 
import torch.nn as nn
import torch.optim as optim
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MyDataset
from model import MyModel
epoch = 100000
hidden_size = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#实例化模型、损失函数、优化器
my_model = MyModel(hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(my_model.parameters(), lr = 1e-4)
# my_model.load_state_dict(torch.load("./result/model.pth"))
# optimizer = optim.SGD(my_model.parameters(), momentum=0.9,lr = 1e-2)
# if os.path.exists("./result/model.pth"):
    # my_model.load_state_dict(torch.load("./result/model.pth"))
    # optimizer.load_state_dict(torch.load("./result/optimizer.pth"))
#开始训练
my_dataset = MyDataset(csv_file = "data.csv")
train_dataloader = DataLoader(dataset = my_dataset, batch_size= 76, shuffle= True, num_workers= 0)
#实时画图
# data = pd.read_csv("data.csv")
# Y_true = data.iloc[:,-1].values
# x = [i+1 for i in range(len(Y_true))]
def train(epoch):
    loss_list =[]
    for i, (data, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        label = label.to(device)
        data = data.to(device)
        output = my_model(data)
        loss = criterion(label,output)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        print('Train Epoch:{} [{}/{}  ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,i*len(data),
            len(train_dataloader.dataset), 100*i/len(train_dataloader), loss.item()))
    avg_loss = np.mean(loss_list)
    torch.save(my_model.state_dict(),"./result/model.pth")
    if epoch % 20 == 0:
        with open("draw_datas/loss.txt","a+") as f:
            f.write("{:.5f}".format(avg_loss) + "\n")
    # with open("draw_datas/accuracy.txt","a+") as f:
    #     f.write("{:.5f}".format(avg_acc) + "\n")
    # torch.save(optimizer.state_dict(),"./result/optimizer.pth")
    # if i % 5 == 0:
    # plt.cla()      
    # plt.scatter(x, Y_true, label = "True", c = "r")
    # plt.plot(x, output.cpu().data.numpy(), c = "b")
    # plt.pause(0.0001)
    
for i in range(epoch):
    train(i)
  

