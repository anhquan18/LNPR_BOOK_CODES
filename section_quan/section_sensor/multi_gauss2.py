import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("sensor_data_200.txt", delimiter=" ", 
            header=None, names = ("date", "time", "ir", "lidar"))
d = data.loc[:,["ir", "lidar"]] # take only ir and lidar value datas

x,y = np.mgrid[280:340, 190:230]
print("x shape:",x.shape,"y shape:", y.shape)

pos = np.empty(x.shape + (2,))
print("before:", pos.shape)

pos[:,:,0] = x
pos[:,:,1] = y

print("after:", pos.shape)

def draw_frequency_distibution():
    sns.jointplot(d["ir"], d["lidar"], d, kind ="kde")
    loc_cov = d.loc[:,["ir", "lidar"]].cov()
    normal_cov = d.cov().values
    print(loc_cov)
    print(normal_cov)

def draw_contour_line():
    ir_lidar = multivariate_normal(mean=d.mean().T, cov=d.cov().values)
    cont = plt.contour(x,y, ir_lidar.pdf(pos))
    ont.clabel(fmt="%1.2e")


if __name__ == "__main__":
    #draw_frequency_distibution()
    draw_contour_line()
    plt.show()
