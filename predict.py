import torch
import torch.nn as nn
from model import MyModel
import  numpy
import pandas as pd 
hidden_size = 11
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('./data/data.csv')
data_nor = data.iloc[:,2:].values
# print(data_nor)
my_model = MyModel(hidden_size).to(device)
my_model.load_state_dict(torch.load("./result/model3.pth"))

def nor(x_list):
    
    std0 = data_nor[:,0].std()
    meanVal0 = data_nor[:,0].mean()
    x_list[0] = (x_list[0] - meanVal0) / std0

    std1 = data_nor[:,1].std()
    meanVal1 = data_nor[:,1].mean()
    x_list[1] = (x_list[1] - meanVal1) / std1

    std2 = data_nor[:,2].std()
    meanVal2 = data_nor[:,2].mean()
    x_list[2] = (x_list[2] - meanVal2) / std2
    # print(meanVal0,meanVal1,meanVal2)
    return x_list

def predict():
    my_model.eval()
    with torch.no_grad():
        while True:
            x_list = []
            print("Enter 'q' to quit!")
            x1 = input("Please input your data(粒径):")
            if x1 == "q":
                break
         
            x2 = input("Please input your data(体积分数):")
            if x2 == "q":
                break
         
            x3 = input("Please input your data(温度):")
            if x3 == "q":
                break
            x_list = [eval(x1),eval(x2),eval(x3)]
            x_list = nor(x_list)
            print(x_list)
            inputs = torch.Tensor(x_list).to(device)
            output = my_model(inputs) 
            output = output.cpu().data.numpy()
            print("The result is {}".format(output))
    
if __name__ == "__main__":
    predict()