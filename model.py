#定义模型
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, hidden_size):
        super(MyModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self,x):
        x = x.view(-1,3)
        out = self.fc(x)
        return out

        