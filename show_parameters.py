import torch
import numpy as np
from model import MyModel

# 声明定义好的线性回归模型
my_model = MyModel(7)
my_model.load_state_dict(torch.load("result/model.pth"))

# print(list(my_model.parameters()))
print(list(my_model.named_parameters()))
