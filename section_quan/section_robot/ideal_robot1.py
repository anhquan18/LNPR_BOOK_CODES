import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np


class World:
    def __init__(self):
        self.objects = []

    def append(self, obj):  # add object to the world
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(8,8))  # 8x8 inch map
        ax = fig.add_subplot(111)        # prepare subplot
        ax.set_aspect('equal')  #横軸比を座標の値と一致する
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_xlabel("X", fontsize=20)
        ax.set_ylabel("Y", fontsize=20)
        
        for obj in self.objects: obj.draw(ax)

        plt.show()

class IdealRobot:
    def __init__(self, pos, color="black"):
        self.pos = pos
        self.r = 0.2
        self.color = color

    def draw(self, ax):
        x, y, theta = self.pos
        xn = x + self.r*math.cos(theta) # robot x corrd
        yn = y + self.r*math.sin(theta)  # robot y corrd
        ax.plot([x, xn], [y, yn], color=self.color) # robot direction
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        ax.add_patch(c)

if __name__ == '__main__':
    world = World()
    #world.draw()

    robot1 = IdealRobot(np.array([2, 3, math.pi/6]).T) 
    robot2 = IdealRobot(np.array([-2, -1, math.pi/5*6]).T, "red")
    world.append(robot1)
    world.append(robot2)
    world.draw()
