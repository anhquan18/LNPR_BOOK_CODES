import sys
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as anm
import numpy as np

x,y,th = 0,1,2


class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []
        self.time_span = time_span
        self.time_interval = time_interval
        self.debug = debug 

    def append(self, obj):  # add object to the world
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(8,8))  # 8x8 inch map
        ax = fig.add_subplot(111)        # prepare subplot
        ax.set_aspect('equal')
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_xlabel("X", fontsize=20)
        ax.set_ylabel("Y", fontsize=20)

        elems = []

        if self.debug:
            for i in range(1000): self.one_step(i, elems, ax) # do nothing when debug is True
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax), frames=int(self.time_span/self.time_interval)+1, interval = int(self.time_interval*1000), repeat=False)
            plt.show()

    def one_step(self, i, elems, ax):
        #print(elems) # debug objects in the world's list
        while elems: elems.pop().remove()
        time_str = "t=%.2f[s]"%(self.time_interval*i) #show time 
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10)) 
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(1.0)


class IdealRobot:
    def __init__(self, pos, agent=None, sensor=None, color="black"):
        self.pos = pos
        self.r = 0.2
        self.color = color
        self.agent = agent
        self.poses = [pos]
        self.sensor = sensor

    def draw(self, ax, elems):
        x, y, theta = self.pos
        xn = x + self.r*math.cos(theta) # robot x corrd
        yn = y + self.r*math.sin(theta)  # robot y corrd
        # ax.plot return a list of matplotlib.lines.Line2D
        elems += ax.plot([x, xn], [y, yn], color = self.color) # elems += add object
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        # ax.add_patch return the matplotlib.patches.Circle object
        elems.append(ax.add_patch(c))

        self.poses.append(self.pos)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color='black')
        if self.sensor:
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agentand hasattr(self.agent, "draw"):
            self.agent.draw(ax,elems)

    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pos) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        self.pos = self.state_transition(nu, omega, time_interval, self.pos)

    @classmethod
    def state_transition(cls,nu,omega, time, pose):
        t0 = pose[th]
        if math.fabs(omega) < 1e-10:
            return pose + np.array( [nu*math.cos(t0), 
                                     nu*math.sin(t0),
                                     omega] ) * time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)),
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time] )


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega


class Map:
    def __init__(self):
        self.landmarks = []

    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks) + 1
        self.landmarks.append(landmark)

    def draw(self, ax, elems):
        for lm in self.landmarks: lm.draw(ax, elems)


class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x,y],).T
        self.id = None

    def draw(self, ax, elems):
        c = ax.scatter(self.pos[x], self.pos[y], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[x], self.pos[y], "id:"+str(self.id), fontsize=10))


class IdealCamera:
    def __init__(self, env_map, 
                 distance_range=(0.5,6.0),
                 direction_range(-math.pi/3, math.pi/3):
        self.map = env_map
        self.lastdata = []

        self.distance_range = distance_range
        self.direction_range = direction_range

    def visible(self, polarpos): #Condition for calculating landmark
        if polarpos is None:
            return False
        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1] \
                and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z)
                observed.append((z, lm.id))
        self.lastdata = observed
        return observed
    
    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x,y,th = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + th)
            ly = y + distance * math.sin(direction + th)
            elems += ax.plot([x,lx], [y,ly], color='pink')

    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2]
        phi = math.atan2(diff[y], diff[x]) - cam_pose[th]
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi += 2*np.pi
        return np.array( [np.hypot(*diff), phi] ).T


if __name__ == '__main__':
    world = World(30, 0.1)

    if len(sys.argv) == 2:
        #go straight with v == 0.1[m/s] from (0,0,0)
        robot_pos = IdealRobot.state_transition(0.1, 0.0, 1.0, np.array([0,0,0]).T)
        print(robot_pos)
        #go straight with v == 0.1[m/s],10[deg/s] from (0,0,0) for 9 sec
        robot_pos = IdealRobot.state_transition(0.1, 10.0/180*math.pi, 9.0, np.array([0,0,0]).T)
        print(robot_pos)
        #go straight with v == 0.1[m/s],10[deg/s] from (0,0,0) for 18 sec
        robot_pos = IdealRobot.state_transition(0.1, 10.0/180*math.pi, 18.0, np.array([0,0,0]).T)
        print(robot_pos)
    else:
        # Create Agent
        straight = Agent(0.2, 0.0)
        circling = Agent(0.2, 10.0/180*math.pi)

        # Create Map
        m = Map()
        m.append_landmark(Landmark(2,-2))
        m.append_landmark(Landmark(-1,-3))
        m.append_landmark(Landmark(3,3))

        #Create Robot
        robot1 = IdealRobot(np.array([2, 3, math.pi/6]).T, 
                                        agent=straight, 
                                        sensor=IdealCamera(m)) 
        robot2 = IdealRobot(np.array([-2, -1, math.pi/5*6]).T, 
                                        agent=circling, 
                                        sensor=IdealCamera(m), color="red")
        #robot3 = IdealRobot(np.array([0, 0, 0]).T, color="blue")

        cam = IdealCamera(m)

        world.append(robot1)
        world.append(robot2)
        #world.append(robot3)
        world.append(m)

        world.draw() # draw animation

        p = cam.data(robot2.pos)
        print(p)
