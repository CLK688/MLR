from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def Draw_2(data_file, result_file):
    data = pd.read_csv(data_file)
    data2 = pd.read_csv(result_file)

    Y_true = data.iloc[:,-1].values
    Y_pre = data2.iloc[:,-1].values
    plt.figure()
    x = [i+1 for i in range(len(Y_true))]
    # plt.ylim(0.5,1.5)
    # plt.scatter(x, Y_true, label = "True", c = "r")
    plt.plot(x, Y_true,label = "true", c = "r")
    plt.plot(x, Y_pre, label = "Pre", c = "b")
    
    plt.show()

Draw_2("data.csv","data_predict.csv")