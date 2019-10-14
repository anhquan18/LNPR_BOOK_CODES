import sys     ###motion_test_header
sys.path.append('../scripts/')
from robot import *
import copy    ###motion_test_linear


world = World(40.0, 0.1)

initial_pose = np.array([0, 0, 0]).T
robots = []
r = Robot(initial_pose, sensor=None, agent=Agent(0.0, 0.1))

for i in range(100):
    copy_r = copy.copy(r)
    copy_r.distance_until_noise = copy_r.noise_pdf.rvs() #最初に雑音が発生するタイミングを変える
    world.append(copy_r)     #worldに登録することでアニメーションの際に動く
    robots.append(copy_r)   #オブジェクトの参照のリストにロボットのオブジェクトを登録

world.draw()

import pandas as pd ###motion_test_stats（下のセルまで。データは上の5行程度を掲載）
poses = pd.DataFrame([ [math.sqrt(r.pose[0]**2 + r.pose[1]**2), r.pose[2]] for r in robots],
                     columns=['r', 'theta'])
#print (poses)

print(poses.mean())
print(poses.var())
print(poses.std())
