'''
2D - 4D 4-Phsae Spacecraft Docking Environment

Created by Anthony Aborizk 
Inspiration: Kyle Dunlap, Kai Delsing, Kerianne Hobbs, R. Scott Erwin, Christopher Jewison

Description:
	A deputy spacecraft is trying to dock with and relocate the chief spacecraft in
    Hill's frame. The mission is broken into 4 phases. The first beginning 10,000 km
    from the target with angles-only measurements and ending at 1,000 km. The second 
    picks up where at 1,000 km where range capable measurements are used. The third 
    begins at the end of the second, 100 km form the target. Spacecraft must enter a 
    cone and commence docking. The 4th and final phase requires the deputy to relocate 
    the target 20,000 km away.  

Observation (Deputy):
	Type: Box(4)
	Num 	Observation 		Min 	Max
	0		x position			-Inf 	+Inf
	1		y position			-Inf 	+Inf
	2		x velocity			-Inf 	+Inf
	3		y velocity			-Inf 	+Inf

Actions (Discrete or Continuous):
	Type: Discrete(9)
	Num 	X Force	 	Y Force	 (Newtons)
	0		-1			-1
	1		-1			 0
	2		-1			 1
	3		 0			-1
	4		 0			 0
	5		 0			 1
	6		 1			-1
	7		 1			 0
	8		 1			 1

	Type: Continuous Box(2,)
	[X direction, Y direction]
	Each value between -1 and +1 Newtons

Reward:
	+1 for reaching the chief 
	-1 for going out of bounds
	-1 for running out of time
    -1 for running out of fuel
    -1 for crashing

Starting State:
	Deputy start 10,000 km away from chief at random angle
	x and y velocity are both between -1.44 and +1.44 m/s

Episode Termination:
	Deputy relocates chief
	Deputy hits chief
	Deputy goes out of bounds
	Out of time
    Out of fuel

Integrators:
	Euler: Simplest and quickest
	Quad: ~4 times slower than Euler
	RK45: ~10 times slower than Quad

Upgrade plan: 
    Starting with a simple docking env (current)
    Will upgrade to phase three only environment (wip)
    Add Phase 2, then 1, then 4 (tbd)
'''

import gym
import math
import random
import numpy as np
from gym import spaces
from scipy import integrate
from gym.utils import seeding
from pyparsing import java_style_comment
from envs.docking.rendering import DockingRender as render


class SpacecraftDocking(gym.Env):

    def __init__(self):

        self.x_chief = 0             # m
        self.y_chief = 0             # m
        self.theta_chief = 0         # rad
        self.position_deputy = 1000  # m (Relative distance from chief)
        self.MASS_DEPUTY = 12        # kg
        self.N = 0.001027            # rad/sec (mean motion)
        self.TAU = 1                 # sec (time step)
        self.integrator = 'Euler'    # Either 'Quad', 'RK45', or 'Euler' (default)
        self.force_magnitude = 1     # Newtons
        # m (In either direction)
        self.x_threshold = 1.5 * self.position_deputy
        # m (In either direction)
        self.y_threshold = 1.5 * self.position_deputy
        # m (|x| and |y| must be less than this to dock)
        self.pos_threshold = .1
        # m/s (Relative velocity must be less than this to dock)
        self.VEL_THRESH = .2
        self.max_time = 4000        # seconds
        self.max_control = 2500     # Newtons
        self.init_velocity = (self.position_deputy + 625) / 1125  # m/s (+/- x and y)
        self.DOF = '3d'                # Degrees of Freedom. 
        #For Tensorboard Plots#
        self.success = 0            # Used to count success rate for an epoch
        self.failure = 0            # Used to count out of bounds rate for an epoch
        self.overtime = 0           # Used to count over max time/control for an epoch
        self.crash = 0              # Used to count crash rate for an epoch

        #Thrust & Particle Variables#
        # what type of thrust visualization to use. 'Particle', 'Block', 'None'
        self.thrustVis = 'None'
        self.particles = []         # list containing particle references
        self.p_obj = []             # list containing particle objects
        self.trans = []             # list containing particle
        self.p_velocity = 20        # velocity of particle
        self.p_ttl = 4              # (steps) time to live per particle
        # (deg) the variation of launch angle (multiply by 2 to get full angle)
        self.p_var = 3

        #Ellipse Variables#
        self.ellipse_a1 = 100       # m
        self.ellipse_a2 = 1000      # m
        self.ellipse_a3 = 10000     # m
        self.ellipse_quality = 150  # 1/x * pi
        self.theta_los = 1.0472/2   # radians

        #Trace Variables#
        self.trace = 1              # (steps)spacing between trace dots
        self.traceMin = True        # sets trace size to 1 (minimum) if true
        self.tracectr = self.trace

        #Noise Measurements
        #account for noise
        self.nn=0
        self.mm=True
        if self.mm==True:
            self.noise=self.nn+random.randint(0,1)
        else:
            self.noise=self.nn        

        #Customization Options
        # gym thing - must be set to show up
        self.viewer = None
        # if set to true, it will print resolution
        self.showRes = False
        # sets the size of the rendering (size of window)
        self.scale_factor = .5 * 500 / self.position_deputy
        # if velocity arrow is shown
        self.velocityArrow = False
        self.forceArrow = False           # if force arrow is shown
        self.LoS = True
        self.bg_color = (0, 0, .15)       # r,g,b
        #color of background (sky)
        self.stars = 40                   # sets number of stars; adding more makes program run slower
        # Set to true to print termination condition
        self.termination_condition = True

        high = np.array([np.finfo(np.float32).max,  # x position (Max possible value +inf)
                         np.finfo(np.float32).max,              # y position
                         np.finfo(np.float32).max,              # x velocity
                         np.finfo(np.float32).max],             # y velocity
                        dtype=np.float32)

        self.action_select()  # Select discrete or continuous action space

        if self.action_type == 'Discrete':  # Discrete action space
            self.action_space = spaces.Discrete(9)
        else:  # Continuous action space
            self.action_space = spaces.Box(np.array([-self.force_magnitude, -self.force_magnitude]), np.array([
                                           self.force_magnitude, self.force_magnitude]), dtype=np.float64)

        # Continuous observation space
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()  # Generate random seed


    def action_select(self):  # Defines action type
        self.action_type = 'Discrete'

    def seed(self, seed=None):  # Sets random seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):  # called before each episode
        self.steps = -1  # Step counter
        self.control_input = 0  # Used to sum total control input for an episode

        # Use random angle to calculate x and y position
        # random angle, starts 10km away
        theta = self.np_random.uniform(low=0, high=2*math.pi)
        self.x_deputy = self.position_deputy*math.cos(theta)  # m
        self.y_deputy = self.position_deputy*math.sin(theta)  # m
        
        # Random x and y velocity
        x_dot = self.np_random.uniform(low=-self.init_velocity, high=self.init_velocity)  # m/s
        y_dot = self.np_random.uniform(low=-self.init_velocity, high=self.init_velocity)  # m/s

        self.rH = self.position_deputy  # m (Relative distance from chief)
        
        if self.DOF == '3rot':
            # Random theta and angular velocity
            self.theta_deputy = self.np_random.uniform(low=0, high=2*math.pi)
            theta_dot = self.np_random.uniform(low=-1, high=1)

            self.state = np.array([self.x_deputy, self.y_deputy, x_dot, y_dot, self.theta_deputy, theta_dot])
        else: 
            self.state = np.array([self.x_deputy, self.y_deputy, x_dot, y_dot])

        # self.angle = np.arctan2(self.x_deputy, self.y_deputy)
        # Define observation state

        # self.state = np.array([self.angle, self.rH, x_dot, y_dot])
        return self.state

    def get_reward(self, observations, actions, obs_old, hstep):
        '''calculates reward in mpc function 

        Args:
            observations (nparray): array of observations
            actions (nparray): array of actions
            hstep (int): mpc horizon step

        Returns:
            rewards this step and done conditions: rew and done
        '''

        if (len(observations.shape) == 1):
            observations = np.expand_dims(observations, axis=0)
            actions = np.expand_dims(actions, axis=0)
            batch_mode = False
        else: 
            batch_mode = True
        if (len(obs_old.shape) == 1):
            obs_old = np.expand_dims(obs_old, axis=0)

        if obs_old[0][0] == None: 
            obs_old = self.state

        if (len(obs_old.shape) == 1):
            obs_old = np.expand_dims(obs_old, axis=0)

        xpos = observations[:, 0]
        ypos = observations[:, 1]  # Observations
        x_dot_obs = observations[:, 2]
        y_dot_obs = observations[:, 3]

        rH = np.linalg.norm([xpos,ypos], axis=0)
        x_force = actions[:, 0]
        y_force = actions[:, 1]
        control_input = (abs(x_force) + abs(y_force)) * self.TAU
        
        if self.DOF == '3rot': 
            theta = observations[:, 4]
            theta_dot = observations[:, 5]
            theta_force = actions[:, 2]

        # for i in range(xpos.shape[0]):

        vH = np.linalg.norm([x_dot_obs, y_dot_obs], axis=0)  # Velocity Magnitude
        vH_max = 2 * self.N * rH + self.VEL_THRESH        # Max Velocity
        vH_min = 1/2 * self.N * rH - self.VEL_THRESH      # Min Velocity

        # check dones conditions
        dones = np.zeros((observations.shape[0],))
        dn1 = np.zeros((observations.shape[0],))
        dn2 = np.zeros((observations.shape[0],))
           
        dn1[abs(ypos) <= self.pos_threshold] = 1
        dn2[abs(xpos) <= self.pos_threshold] = 1
        dones[dn1+dn2>1] = 1 # if s/c is out of bounds
        dones[abs(xpos) > self.x_threshold] = 1
        dones[abs(ypos) > self.y_threshold] = 1
        # dones[control_input > self.max_control] = 1
        dones[(self.steps+hstep) * self.TAU >= self.max_time-1] = 1

        #calc rewards
        # reward = np.zeros((observations.shape[0],))
        failure = np.zeros((observations.shape[0],))    
        success = np.zeros((observations.shape[0],))    
        crash = np.zeros((observations.shape[0],))    
        overtime = np.zeros((observations.shape[0],))    

        # if not dones: 
        rH_old = np.linalg.norm([obs_old[:,0],obs_old[:,1]], axis=0)
        c = np.array([0, 100, 0])
        position = [xpos, ypos, 0]
        val = np.dot(position, c)/ (np.linalg.norm(c, axis=0) * rH)
        th = np.arctan2(ypos,xpos) 

        reward = (-1 - rH + rH_old)/2000 * self.TAU
        if ~all(dones):  
            # reward[dones==0] -= rH[dones==0]/100000
            reward[((dones==0) & (vH < vH_min))] += -0.0075*abs(vH[((dones==0) & (vH < vH_min))]-vH_min[((dones==0) & (vH < vH_min))]) * self.TAU
            reward[((dones==0) & (vH > vH_max))] += -0.0035*abs(vH[((dones==0) & (vH > vH_max))]-vH_max[((dones==0) & (vH > vH_max))]) * self.TAU
            reward[((dones==0) & (vH < 2*self.VEL_THRESH) & (vH < vH_min))] += -0.0075/2 * self.TAU
            reward[((dones==0) & (vH < 2*self.VEL_THRESH) & (vH > vH_max))] += -0.0075/2 * self.TAU
            reward[((dones==0) & (val >= np.cos(th/2)))] += 0.1

        # elif: 
        if all(dones) != False: 
            reward[((dones==1) & (abs(xpos) <= self.pos_threshold) & (abs(ypos) <= self.pos_threshold) & (vH > self.VEL_THRESH))] += -0.001
            reward[((dones==1) & (abs(xpos) <= self.pos_threshold) & (abs(ypos) <= self.pos_threshold) & (vH <= self.VEL_THRESH))] += 1
            reward[((dones==1) & ((self.steps+hstep) * self.TAU > self.max_time))] += -1
            reward[((dones==1) & (abs(xpos) > self.pos_threshold) & (abs(ypos) > self.pos_threshold))] += -1

            success[((dones==1) & (abs(xpos) <= self.pos_threshold) & (abs(ypos) <= self.pos_threshold) & (vH <= self.VEL_THRESH))] = 1
            crash[((dones==1) & (abs(xpos) <= self.pos_threshold) & (abs(ypos) <= self.pos_threshold) & (vH > self.VEL_THRESH))] = 1
            failure[((dones==1) & (abs(xpos) > self.pos_threshold) & (abs(ypos) > self.pos_threshold))] = 1
            overtime[((dones==1) & ((self.steps+hstep) * self.TAU >= self.max_time-1))] = 1

        info = {}
        info['success'] = sum(success)
        info['crash'] = sum(crash)
        info['failure'] = sum(failure)
        info['overtime'] = sum(overtime)


        return reward, dones, info

    def step(self, action):
        self.steps += 1  # step counter

        if self.action_type == 'Discrete':
            # Stop program if invalid action is used
            assert self.action_space.contains(action), "Invalid action"
        else:
            # Clip action to be within boundaries - only for continuous
            action = np.clip(action, -self.force_magnitude,self.force_magnitude)

        # Extract current state data
        if self.DOF == '3rot':
            x, y, x_dot, y_dot, theta, theta_dot = self.state
            self.x_force, self.y_force, self.theta_force = action

        else: 
            x, y, x_dot, y_dot = self.state
            self.x_force, self.y_force = action

        # Add total force for given time period
        # self.control_input += (abs(self.x_force) + abs(self.y_force)) * self.TAU

        # Integrate Acceleration and Velocity
        if self.integrator == 'RK45':  # Runge-Kutta Integrator
            # Define acceleration functions
            def x_acc_int(t, x):
                return (3 * self.N ** 2 * x) + (2 * self.N * y_dot) + (self.x_force / self.MASS_DEPUTY)

            def y_acc_int(t, y):
                return (-2 * self.N * x_dot) + (self.y_force / self.MASS_DEPUTY)
            # Integrate acceleration to calculate velocity
            x_dot = integrate.solve_ivp(
                x_acc_int, (0, self.TAU), [x_dot]).y[-1, -1]
            y_dot = integrate.solve_ivp(
                y_acc_int, (0, self.TAU), [y_dot]).y[-1, -1]
            # Define velocity functions

            def vel_int(t, x):
                return x_dot, y_dot
            # Integrate velocity to calculate position
            xtemp, ytemp = integrate.solve_ivp(
                vel_int, (0, self.TAU), [x, y]).y
            x = xtemp[-1]
            y = ytemp[-1]

        elif self.integrator == 'Euler':  # Simple Euler Integrator
            # Define acceleration functions
            x_acc = (3 * self.N ** 2 * x) + (2 * self.N * y_dot) + \
                (self.x_force / self.MASS_DEPUTY)
            y_acc = (-2 * self.N * x_dot) + (self.y_force / self.MASS_DEPUTY)
            # Integrate acceleration to calculate velocity
            x_dot = x_dot + x_acc * self.TAU
            y_dot = y_dot + y_acc * self.TAU
            # Integrate velocity to calculate position
            x = x + x_dot * self.TAU
            y = y + y_dot * self.TAU

        else:  # Default 'Quad' Integrator
            # Integrate acceleration to calculate velocity
            x_dot = x_dot + integrate.quad(lambda x: (3 * self.N ** 2 * x) + (
                2 * self.N * y_dot) + (self.x_force / self.MASS_DEPUTY), 0, self.TAU)[0]
            y_dot = y_dot + integrate.quad(lambda y: (-2 * self.N * x_dot) + (
                self.y_force / self.MASS_DEPUTY), 0, self.TAU)[0]
            # Integrate velocity to calculate position
            x = x + integrate.quad(lambda x: x_dot, 0, self.TAU)[0]
            y = y + integrate.quad(lambda y: y_dot, 0, self.TAU)[0]

        # Define new observation state
        observation = np.array([x, y, x_dot, y_dot])

        # Relative distance between deputy and chief (assume chief is always at origin)
        self.rH = np.sqrt(x**2 + y**2)
        self.vH = np.sqrt(x_dot**2 + y_dot**2)  # Velocity Magnitude
        self.vH_max = 2 * self.N * self.rH + self.VEL_THRESH  # Max Velocity
        self.vH_min = 1/2 * self.N * self.rH - self.VEL_THRESH  # Min Velocity
        rew, done, reward = self.get_reward(observation, action, self.state, 0)
        self.state = observation
        reward['rew'] = rew

        return self.state, reward, done, {}

    # Used to check if velocity is over max velocity constraint
    def check_velocity(self, x_force, y_force):
        # Extract current state data
        x, y, x_dot, y_dot = self.state
        # Define acceleration functions
        x_acc = (3 * self.N ** 2 * x) + (2 * self.N * y_dot) + \
            (x_force / self.MASS_DEPUTY)
        y_acc = (-2 * self.N * x_dot) + (y_force / self.MASS_DEPUTY)
        # Integrate acceleration to calculate velocity
        x_dot = x_dot + x_acc * self.TAU
        y_dot = y_dot + y_acc * self.TAU

        # Check if over max velocity, and return True if it is violating constraint
        rH = np.sqrt(x**2 + y**2)  # m, distance between deputy and chief
        vH = np.sqrt(x_dot**2 + y_dot**2)  # Velocity Magnitude
        vH_max = 2 * self.N * self.rH + self.VEL_THRESH  # Max Velocity # Max Velocity

        # If violating, return True
        if vH > vH_max:
            value = True
        else:
            value = False

        # Calculate velocity angle
        vtheta = math.atan(y_dot/x_dot)
        if x_dot < 0:
            vtheta += math.pi

        return value, vH_max, vtheta

     # Rendering Functions
    def render(self, mode):
        render.renderSim(self, mode='human')

    def close(self):
        render.close(self)


    def predict(self, obs, action):
        # self.steps += 1  # step counter
        action = np.clip(action, -self.force_magnitude,self.force_magnitude)

        # Extract current state data
        x, y, x_dot, y_dot = obs.transpose()

        x_force, y_force = action.transpose()

        # Define acceleration functions
        x_acc = (3 * self.N ** 2 * x) + (2 * self.N * y_dot) + \
            (x_force / self.MASS_DEPUTY)
        y_acc = (-2 * self.N * x_dot) + (y_force / self.MASS_DEPUTY)
        # Integrate acceleration to calculate velocity
        x_dot = x_dot + x_acc * self.TAU
        y_dot = y_dot + y_acc * self.TAU
        # Integrate velocity to calculate position
        x = x + x_dot * self.TAU
        y = y + y_dot * self.TAU

        # Define new observation state
        next_state = np.array([x, y, x_dot, y_dot])

        return next_state.transpose()

# Used to define 'spacecraft-docking-continuous-v0'

class SpacecraftDockingContinuous(SpacecraftDocking):
    def action_select(self):  # Defines continuous action space
        self.action_type = 'Continuous'
