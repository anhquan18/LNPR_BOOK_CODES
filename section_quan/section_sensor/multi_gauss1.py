import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal #多次元ガウス分布


data = pd.read_csv("sensor_data_700.txt", delimiter=" ",header=None, names = ("date", "time", "ir", "lidar"))

d = data[(data["time"] < 160000) & (data["time"] >= 120000)] # 12:00 -> 16:00 のデータだけを抽出
d = d.loc[:, ["ir", "lidar"]]

c = d.cov().values + np.array([[0,-25], [-25, 0]]) ###covadd###
#print(c)


def multi_gauss():
    sns.jointplot(d["ir"], d["lidar"], d, kind="kde")

def cal_covar():
    #print("light sensor's mesurement variance:", d.ir.var())
    #print("LiDAR sensor's mesurement variance:", d.lidar.var())

    #print(d.mean().shape)  # mean 
    #print(d.mean().T.shape)

    diff_ir = d.ir - d.ir.mean()
    diff_lidar = d.lidar - d.lidar.mean()

    print(diff_ir)

    #print("d ir:\n", d.ir)

    a = diff_ir * diff_lidar
    #print ("Covariance:", sum(a)/ (len(d) - 1))

    #print(type(d.mean()))
    #print(type(d.cov().values))

    #print (d.cov())  # covariance

def draw_multi_var_graph():
    #ir_lidar = multivariate_normal(mean = d.mean().values, cov = d.cov().values)
    ir_lidar = multivariate_normal(mean = d.mean(), cov = d.cov())
    tmp = multivariate_normal(mean=d.mean().values, cov=c)
    x, y = np.mgrid[0:40, 710:750] # 2 dimensions sensor value x as lidar, y as ir

    pos = np.empty(x.shape+(2,))   # x is 40 * 40 2 dimensions list -> create 3 dimensions list from 2 dimensions matrx

    pos[:, :, 0] = x 
    pos[:, :, 1] = y

    #cont = plt.contour(x, y, ir_lidar.pdf(pos)) #calculate density from x, y corrdinate 
    cont = plt.contour(x, y, tmp.pdf(pos))
    cont.clabel(fmt='%1.1e') 

if __name__ == '__main__':
    cal_covar()
    #draw_multi_var_graph()
    plt.show()
