import sys
sys.path.append('../scripts')
from robot import *
from mcl1 import *

if __name__ == "__main__":
    time_interval = 0.1
    world = World(45, 0.1, debug=False)
    motion_noise_stds = {"nn":0.19, "no":0.1, "on":0.23, "oo":0.22}
    initial_pose = np.array([0,3, math.pi*3/2]).T
    agents = []

    m = Map()
    for lm in [(-4, 4, 0.0*math.pi, 'red'), (3,-3, -math.pi, 'blue')]: m.append_landmark(RobotLandMark(*lm))
    world.append(m)

    estimator = Mcl(m, initial_pose, 50, motion_noise_stds)
    circling = EstimationAgent(time_interval, 0.2, 0.0/180*math.pi, estimator)
    forward_and_turn = EstimationAgent(time_interval, 0.2, 0.0*math.pi/180, estimator)

    agents.append(circling)
    agents.append(forward_and_turn)

    r = Robot(initial_pose, sensor=Camera(m), agent=agents[1], color="black")
    world.append(r)

    world.draw()

