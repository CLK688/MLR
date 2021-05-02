import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

loss_list = []
with open("draw_datas/loss.txt", "r") as f:
    for line in f.readlines():
        line = line.strip().split(" ")   #按照空格进行切分
        loss_list.append(line)   
font1 = {'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 10 }
x_loss = [(i+1) * 20 for i in range(len(loss_list))]
def draw():
    plt.figure("Loss")
    #生产画布
    plt.gcf().set_facecolor(np.ones(3) * 240 / 255)
    #生成网格 plt.grid(b,which,axis,color,linestyle,linewidth,**kwarges)
    #axis取值为"both","x"沿着x轴画线；"y"沿着y轴画线
    #color网格的颜色
    #linestyle :网格的线条形状 "-."等
    plt.grid(axis="y") #生成网格
    plt.scatter(x_loss, loss_list, marker= ".", label = "Avg_loss", c = "b")
    plt.title("Loss",fontsize = 13)
    plt.xlabel("Epoch number")
    plt.ylabel("Avg_loss")
    plt.legend()
    plt.savefig("draw_datas/Loss.jpg",bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__":
    draw()