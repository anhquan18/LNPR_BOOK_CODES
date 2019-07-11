import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import math

x, y = np.mgrid[0:200, 0:100]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

a = multivariate_normal(mean=[50, 50], cov=[[50,0],[0,100]])
b = multivariate_normal(mean=[100, 50], cov=[[125,0],[0,25]])
c = multivariate_normal(mean=[150, 50], cov=[[100,-25*math.sqrt(3)],[-25*math.sqrt(3),50]])

eig_vals, eig_vec = np.linalg.eig(c.cov) ###eigen###

V = eig_vec # eig_vec == 固有ベクトル行列
L = np.diag(eig_vals) # np.diagで対角行列を作成

print("分析した物の計算\n", V.dot(L.dot(np.linalg.inv(V))))
print("元の共分散行列:\n", np.array([[100, -25*math.sqrt(3)], [-25*math.sqrt(3), 50]]))
print("\n")

def draw_multi_dis():
    for dis in [a,b,c]:
        plt.contour(x, y, dis.pdf(pos))

    plt.gca().set_aspect('equal') # gca: return object manage axes
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('y')

def output_eig_vals():
    print("eig_vals:\n", eig_vals)
    print("eig_vectors:\n", eig_vec)
    print("固有ベクトル1:", eig_vec[:,0]) #eig_vecの縦の列が固有ベクトルに対応
    print("固有ベクトル2:", eig_vec[:,1])

def draw_vector_with_multi_var():
    plt.contour(x, y, c.pdf(pos))

    v = 2*math.sqrt(eig_vals[0])*eig_vec[:,0] # double the size of the vector because it too small
    print("v[0]:", v)
    plt.quiver(c.mean[0], c.mean[1], v[0], v[1], color='red', angles='xy', scale_units='xy', scale=1)

    v = 2*math.sqrt(eig_vals[1])*eig_vec[:,1] # vector length equal to the length of variance(分散)
    print("v[1]:", v)
    plt.quiver(c.mean[0], c.mean[1], v[0], v[1], color='blue', angles='xy', scale_units='xy', scale=1)

    plt.gca().set_aspect('equal')


if __name__ == '__main__':
    #draw_multi_dis()
    output_eig_vals()
    draw_vector_with_multi_var()
    plt.show()
