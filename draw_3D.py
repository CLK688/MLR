#画图
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def Draw(data_file, result_file):
    data = pd.read_csv(data_file)
    data2 = pd.read_csv(result_file)
    x1 = data.iloc[:,2].values
    x2 = data.iloc[:,3].values
    Y_true = data.iloc[:,-1].values
    Y_pre = data2.iloc[:,-1].values
    fig = plt.figure()
    ax1 = fig.add_subplot(221,projection='3d')  #这种方法也可以画多个子图
    ax2 = fig.add_subplot(222,projection='3d')  #这种方法也可以画多个子图
    ax3 = fig.add_subplot(223,projection='3d')  #这种方法也可以画多个子图
    ax4 = fig.add_subplot(224,projection='3d')  #这种方法也可以画多个子图
    
    # plt.xlim(0,1) #设置x轴的坐标范围
    # plt.ylim(0,1) #设置y轴的坐标范围
    # plt.zlim(0,1)
    ax1.scatter(x1, x2, Y_true, c='skyblue', s=60)
    ax2.scatter(x1, x2, Y_pre,  c='skyblue', s=60)
    ax3.plot_trisurf(x1, x2, Y_true, cmap='rainbow', linewidth=0.01)
    ax4.plot_trisurf(x1, x2, Y_pre,  cmap='rainbow', linewidth=0.01)
    ax1.set_zlim(0, 1.2)
    ax2.set_zlim(0, 1.2)
    ax3.set_zlim(0, 1.2)
    ax4.set_zlim(0, 1.2)
    plt.show()

Draw("data.csv","data_predict.csv")