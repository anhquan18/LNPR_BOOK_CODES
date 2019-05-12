import sys 
sys.path.append('../scripts/')
from robot import *
from scipy.stats import multivariate_normal


class Particle:  ###beforeimpllikelihood
    def __init__(self, init_pose, weight):
        self.pose = init_pose
        self.weight = weight

    def motion_update(self, nu, omega, time, noise_rate_pdf):
        ns = noise_rate_pdf.rvs()
        pnu = nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)
        pomega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
        self.pose = IdealRobot.state_transition(pnu, pomega, time, self.pose)

    def observation_update(self, observation, envmap, distance_dev_rate, direction_dev):  #changed
        print(observation)

class Mcl:  ###recvmapmcl
    def __init__(self, envmap, init_pose, num, motion_noise_stds, \
                 distance_dev_rate=0.14, direction_dev=0.05):    
        self.particles = [Particle(init_pose, 1.0/num) for i in range(num)]
        self.map = envmap  
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev

        v = motion_noise_stds
        c = np.diag([v["nn"]**2, v["no"]**2, v["on"]**2, v["oo"]**2])
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)
        
    def motion_update(self, nu, omega, time):
        for p in self.particles: p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)
            
    def observation_update(self, observation): 
        for p in self.particles: p.observation_update(observation, self.map, \
                                                      self.distance_dev_rate, self.direction_dev) 
        
    def draw(self, ax, elems):      
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2])*p.weight*len(self.particles) for p in self.particles] 
        vys = [math.sin(p.pose[2])*p.weight*len(self.particles) for p in self.particles]  
        elems.append(ax.quiver(xs, ys, vxs, vys, \
                               angles='xy', scale_units='xy', scale=1.5, color="blue", alpha=0.5)) 

class MclAgent(Agent):  ###recvmap 
    def __init__(self, time_interval, nu, omega, particle_pose, envmap, particle_num=100, \
                motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2}): 
        Agent.__init__(self,nu, omega)
        self.pf = Mcl(envmap, particle_pose, particle_num, motion_noise_stds) 
        self.time_interval = time_interval
        
        self.prev_nu = 0.0
        self.prev_omega = 0.0
        
    def decision(self, observation=None): 
        self.pf.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.pf.observation_update(observation)
        return self.nu, self.omega
        
    def draw(self, ax, elems):
        self.pf.draw(ax, elems)

if __name__ == '__main__':          
    time_interval = 0.1
    world = World(30, time_interval) 

    ### Create Map with 3 landmark
    m = Map()                                  
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)          

    ### Create Robot###
    circling = MclAgent(time_interval, 0.2, 10.0/180*math.pi, np.array([0, 0, 0]).T, m) #give map as parameter
    r = Robot(np.array([0,0,0]).T, sensor=Camera(m), agent=circling, color="red")
    world.append(r)

    world.draw()                       # Animation # 
   # r.one_step(time_interval)  # debug when no animation
