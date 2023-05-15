'''
Autonomous Rendezvous, Proximity Operations, and Docking (ARPOD)
2D 4-Phase Spacecraft Docking Environment
Created by Anthony Aborizk 
Citations/Inspiration: Kyle Dunlap, Kai Delsing, Kerianne Hobbs, R. Scott Erwin, Christopher Jewison
Description:
	A deputy spacecraft is trying to dock with and relocate a chief spacecraft in
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
Actions (Continuous):
	Type: Continuous Box(2,)
	[X direction, Y direction]
	Each value between -1 and +1 Newtons
Reward:
	+1 for reaching the chief 
	-1 for going out of bounds
	-1 for running out of time
    -1 for running out of fuel
    -1 for crashing
    #todo but wait, there's more!
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
    Starting with a simple docking env (done 2022) 
    Will upgrade to phase three only environment (done 2022)
    Will add phase 2 and enable switching between the tww (done 2022)
    Add Phase 4 (wip)
    Add Phase 1 (wip)
    Add verious degrees of freedom options (wip)
    Add noise (wip)
Addition of DOF
    The options available for degrees of freedom include: 
    - 2D = 2D translational motion (default)
    - 3D = 3D translational motion 
    - 3Drot = 2D translation + orientation about z(or 'k')-axis
    - 6D = 3D translation + 3D rotation
'''

import os
import gymnasium as gym
import math
import time
import pickle
import random
import numpy as np
from gymnasium.spaces import Box
from gymnasium.utils import seeding
from pyparsing import java_style_comment
from envs.docking.rendering import DockingRender as render
from gymnasium.wrappers import EnvCompatibility

## @EnvCompatibility
class SpacecraftDockingContinuous(gym.Env):

    def __init__(self, logdir=None):
        '''
        Some of these initial conditions are taken from  "A Spacecraft
        Benchmark Problem for Hybrid Control and Estimation" by 
        Jewison and Erwin
        '''
        self.x_chief = 0             # m
        self.y_chief = 0             # m
        self.position_deputy = 100   # m (Relative distance from chief)
        self.MASS_DEPUTY = 12        # kg 
        self.MASS_TARGET = 200      # kg
        a = 42164000.                # semi-major axis of GEO in m
        mu = 3.986e14                # Earth's gravitaitonal constant m^3/s^2
        self.N = math.sqrt(mu/a**3)  # rad/sec (mean motion) (circular orbit)
        self.TAU = 1                 # sec (time step)
        self.integrator = 'Euler'    # Either 'Quad', 'RK45', or 'Euler' (default)
        self.force_magnitude = 1     # Newtons
        self.torque_mag = 2*math.pi/180# rad
        self.inertia_zz = 0.056      # kg-m**2 (1/2 M*R**2)

        if self.position_deputy > 100: 
            self.phase = 2
        else: 
            self.phase = 3

        # m (In either direction)
        self.x_threshold = 1.5 * self.position_deputy

        # m (In either direction)
        self.y_threshold = 1.5 * self.position_deputy
        
        # m (|x| and |y| must be less than this to dock)
        self.pos_threshold = .1

        # rad (Relative angle must be less than this to dock)
        self.psi_threshold = 0.25   

        # m/s (Relative velocity must be less than this to dock)
        self.VEL_THRESH = .2        # m/s
        self.max_time = 4*60*60     # seconds
        self.max_control = 2500     # Newtons
        self.init_velocity = (self.position_deputy + 625) / 1125  # m/s (+/- x and y)
        self.init_psi = 2*math.pi/180         # angular velocity within [-2, 2] deg/s
        self.DOF = '3Drot'          # Degrees of Freedom. 

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

        if logdir: 
            self.log(None, logdir, True)

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
        self.scale_factor = .6 * 500 / self.position_deputy

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
                         np.finfo(np.float32).max,  # y position
                         2*np.pi,                   # orientation
                         np.finfo(np.float32).max,  # x velocity
                         np.finfo(np.float32).max,  # y velocity
                         np.finfo(np.float32).max], # angular velocity
                        dtype=np.float32)
        low = -high
        low[2] = 0

        self.action_space = Box(np.array([-self.force_magnitude, -self.force_magnitude, \
                                                 -self.torque_mag]), np.array([self.force_magnitude,\
                                                  self.force_magnitude, self.torque_mag]), dtype=np.float32)

        # Continuous observation space
        self.observation_space = Box(low, high, dtype=np.float32)

        self.seed()  # Generate random seed

    def seed(self, seed=None):  # Sets random seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):  # called before each episode
        self.steps = -1         # Step counter
        self.control_input = 0  # Used to sum total control input for an episode

        # Use random angle to calculate x and y position
        if self.phase == 3: 
            # instantiate within line of sight cone constraints
            theta = self.np_random.uniform(low=5*math.pi/6, high=7*math.pi/6)
        else: 
            theta = self.np_random.uniform(low=0, high=2*math.pi)

        self.x_deputy = self.position_deputy*math.cos(theta)  # m
        self.y_deputy = self.position_deputy*math.sin(theta)  # m
        self.psi = self.np_random.uniform(low=0, high=2*math.pi)
        
        # Random x and y velocity
        x_dot = self.np_random.uniform(low=-self.init_velocity, high=self.init_velocity)  # m/s
        y_dot = self.np_random.uniform(low=-self.init_velocity, high=self.init_velocity)  # m/s
        psi_dot = self.np_random.uniform(low=-self.init_psi, high=self.init_psi)
        # (Relative distance from chief)
        self.rH = self.position_deputy  # m 

        # orientation of chaser
        self.state = np.array([self.x_deputy, self.y_deputy,self.psi, x_dot, y_dot, psi_dot])

        empty_dict = dict()  # Added for gymnasium compatibility 
        return self.state, empty_dict
    
       

    def get_reward(self, obs, actions, obs_old, hstep):
        '''calculates reward 
        Args:
            observations (nparray): array of observations
            actions (nparray): array of actions
            hstep (int): mpc horizon step
        Returns:
            rewards this step and done conditions: rew and done
        '''
        if (len(obs.shape) == 1):
            obs = np.expand_dims(obs, axis=0)
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

        xpos        = obs[:, 0]#
        ypos        = obs[:, 1] #
        psi         = obs[:, 2]  # 
                                  # Observations
        x_dot_obs   = obs[:, 3]  #
        y_dot_obs   = obs[:, 4] #
        psi_dot_obs = obs[:, 5]#

        rH = np.linalg.norm([xpos,ypos], axis=0)
        if hstep == 1 or hstep == 0: 
            self.hcinput = self.control_input
        x_force = actions[:, 0]
        y_force = actions[:, 1]
        self.hcinput += (abs(x_force) + abs(y_force)) * self.TAU
        
        vH = np.linalg.norm([x_dot_obs, y_dot_obs], axis=0)  # Velocity Magnitude
        vH_max = 2 * self.N * rH + self.VEL_THRESH        # Max Velocity
        vH_min = 1/2 * self.N * rH - self.VEL_THRESH      # Min Velocity

        rH_old = np.linalg.norm([obs_old[:,0],obs_old[:,1]], axis=0)
        val      = xpos * -1 / (1 * rH)  # dot(position, c) / mag(position)*mag(c)
        # check dones conditions
        
        # check dones conditions
        dones = np.zeros((obs.shape[0],))   # initialize dones vector
        dn1 = np.zeros((obs.shape[0],))
        dn2 = np.zeros((obs.shape[0],))
        dn3 = np.zeros((obs.shape[0],))

        dn1[abs(ypos) <= self.pos_threshold] = 1
        dn2[abs(xpos) <= self.pos_threshold] = 1

        dones[dn1+dn2>1] = 1 # if s/c is within docking bounds
        dones[abs(xpos) > self.x_threshold] = 1  # out of bounds 
        dones[abs(ypos) > self.y_threshold] = 1  # out of bounds 
        dones[self.control_input > self.max_control] = 1
        dones[(self.steps+hstep) * self.TAU >= self.max_time-1] = 1

        #calc rewards
        failure  = np.zeros((obs.shape[0],))   
        success  = np.zeros((obs.shape[0],))    
        crash    = np.zeros((obs.shape[0],))    
        overtime = np.zeros((obs.shape[0],))    
        nofuel   = np.zeros((obs.shape[0],))

        reward = (-1 - rH + rH_old)/2000 * self.TAU # reward for getting closer to target
        reward +=-(psi-0)**2/3000 * self.TAU 

        if ~all(dones):    
            # stay within vel const
            reward[((dones==0) & (vH < vH_min))] += -0.0075*abs(vH[((dones==0) & (vH < vH_min))]-vH_min[((dones==0) & (vH < vH_min))]) * self.TAU 
            reward[((dones==0) & (vH > vH_max))] += -0.0035*abs(vH[((dones==0) & (vH > vH_max))]-vH_max[((dones==0) & (vH > vH_max))]) * self.TAU
            reward[((dones==0) & (vH < 2*self.VEL_THRESH) & (vH < vH_min))] += -0.0075/2 * self.TAU
            reward[((dones==0) & (vH < 2*self.VEL_THRESH) & (vH > vH_max))] += -0.0075/2 * self.TAU

            reward[((dones==0) & (abs(psi_dot_obs) > 1))] += -0.00075/2 * self.TAU
            #? if not done, within 100 m, not inside LoS and not within docking region
            reward[((dones==0) & (rH <= 100) & (val < math.cos(self.theta_los)) & (abs(xpos) > self.pos_threshold) & (abs(ypos) > self.pos_threshold))] += -10
            # reward[((dones==0) & (psi >= self.psi_threshold))] += -.0001

        # if all(dones) != False: 
        reward[( (abs(xpos) <= self.pos_threshold) & (abs(ypos) <= self.pos_threshold) & (vH > self.VEL_THRESH))] += -0.001
        reward[( (abs(xpos) <= self.pos_threshold) & (abs(ypos) <= self.pos_threshold) & (vH <= self.VEL_THRESH))] += 1
        reward[( ((self.steps+hstep) * self.TAU > self.max_time))] += -1
        reward[( (abs(xpos) > self.x_threshold) | (abs(ypos) > self.y_threshold))] += -1 
        # reward[((dones==1)  & (self.hcinput > np.ones((obs.shape[0],))*self.max_control))] = 1
        
        success[((dones==1) & (abs(xpos) <= self.pos_threshold) & (abs(ypos) <= self.pos_threshold) & (vH <= self.VEL_THRESH))] = 1
        crash[((dones==1) & (abs(xpos) <= self.pos_threshold) & (abs(ypos) <= self.pos_threshold) & (vH > self.VEL_THRESH))] = 1
        failure[((dones==1) & (abs(xpos) > self.x_threshold) | (abs(ypos) > self.y_threshold))] = 1
        overtime[((dones==1) & ((self.steps+hstep) * self.TAU >= self.max_time-1))] = 1
        # nofuel[((dones==1)  & (self.hcinput > np.ones((obs.shape[0],))*self.max_control))] = 1

        info = {}
        info['success'] = sum(success)
        info['crash']   = sum(crash)
        info['failure'] = sum(failure)
        info['overtime']= sum(overtime)

        return reward, dones, info

    def step(self, action):
        self.steps += 1  # step counter

        # Clip action to be within boundaries
        action = np.clip(action, [-self.force_magnitude, -self.force_magnitude, -self.torque_mag],\
                                 [self.force_magnitude, self.force_magnitude, self.torque_mag])
        # action[-1] = np.clip(action[-1], -self.torque_mag, self.torque_mag)

        # Extract current state data
        x, y, psi, x_dot, y_dot, psi_dot = self.state
        self.x_force, self.y_force, self.torque = action

        # Add total force for given time period
        self.control_input += (abs(self.x_force) + abs(self.y_force)) * self.TAU

        # Integrate Acceleration and Velocity
        # Define acceleration functions

        x_acc = (3 * self.N ** 2 * x) + (2 * self.N * y_dot) + \
            (self.x_force * np.cos(psi) + self.y_force*np.sin(psi))/ self.MASS_DEPUTY
        y_acc = (-2 * self.N * x_dot) + (-self.x_force*np.sin(psi) + self.y_force * np.cos(psi)) / self.MASS_DEPUTY
        psi_acc = self.torque/self.inertia_zz 
        #* z_acc = -self.N**2*z + self.z_force/self.MASS_chaser

        # Integrate acceleration to calculate velocity
        x_dot = x_dot + x_acc * self.TAU
        y_dot = y_dot + y_acc * self.TAU
        psi_dot = psi_dot + psi_acc * self.TAU
        #* z_dot = z_dot + z_acc * self.TAU

        # Integrate velocity to calculate position
        x = x + x_dot * self.TAU
        y = y + y_dot * self.TAU
        psi_unwrapped = psi + psi_dot * self.TAU
        psi = (psi_unwrapped + np.pi) % (2 * np.pi) # - np.pi
        #* z = z + z_dot * self.TAU

        # Define new observation state
        observation = np.array([x, y, psi, x_dot, y_dot, psi_dot])

        # Relative distance between deputy and chief (assume chief is always at origin)
        self.rH = np.sqrt(x**2 + y**2)

        self.vH = np.sqrt(x_dot**2 + y_dot**2)  # Velocity Magnitude
        self.vH_max = 2 * self.N * self.rH + self.VEL_THRESH  # Max Velocity
        self.vH_min = 1/2 * self.N * self.rH - self.VEL_THRESH  # Min Velocity
        rew, done, reward= self.get_reward(observation, action, self.state, 0)  # What's up with rew and reward??
        self.state = observation
        reward['rew'] = rew
        # reward['rew'] = rew
        if self._logdir:
            self.log(action)

        if done and self._logdir: 
            self.log(action, done=done)

        info = done
        if all(done):
            done = True
            truncated = True
        else:
            done = False
            truncated = False

        return self.state, rew.item(), done, truncated, {} # Keep {} -- empty dict

     # Rendering Functions
    def render(self, mode='human'):
        render.renderSim(self, mode)

    def close(self):
        render.close(self)

    def predict(self, obs, action):
        # self.steps += 1  # step counter
        action = np.clip(action, [-self.force_magnitude, -self.force_magnitude, -self.torque_mag],\
                                 [self.force_magnitude, self.force_magnitude, self.torque_mag])

        # Extract current state data
        x, y,psi, x_dot, y_dot, psi_dot = obs.transpose()

        x_force, y_force, torque = action.transpose()

        # Define acceleration functions
        x_acc = (3 * self.N ** 2 * x) + (2 * self.N * y_dot) + \
            (x_force * np.cos(psi) + y_force*np.sin(psi)) / self.MASS_DEPUTY
        y_acc = (-2 * self.N * x_dot) + (-x_force*np.sin(psi) + y_force * np.cos(psi))/ self.MASS_DEPUTY
        psi_acc = torque/self.inertia_zz 

        # Integrate acceleration to calculate velocity
        x_dot = x_dot + x_acc * self.TAU
        y_dot = y_dot + y_acc * self.TAU
        psi_dot = psi_dot + psi_acc * self.TAU

        # Integrate velocity to calculate position
        x = x + x_dot * self.TAU
        y = y + y_dot * self.TAU
        psi = psi + psi_dot * self.TAU

        # Define new observation state
        next_state = np.array([x, y, psi, x_dot, y_dot, psi_dot])

        return next_state.transpose()

    def log(self, action, data_path=None, initialize=False, done=False):
        '''
        loggs observations also logs actions. '''
        
        if initialize: 
            ##################################
            # CREATE DIRECTORY FOR LOGGING
            ##################################
            # data_folder = 'arpod_logs'
            # data_path = os.path.join(logdir, data_file)

            # if not (os.path.exists(data_path)):
            #     os.makedirs(data_path)

            logname = 'log' +'_'+ time.strftime("%m-%d-%Y_%H-%M")
            self._logdir = os.path.join(data_path, logname)

            if not(os.path.exists(self._logdir)):
                with open(self._logdir, 'w') as fp:
                # use this line of code if you want to make a directory instead of a file
                # os.makedirs(self._logdir)
                    pass

            self._logdata = np.array([])
          
            return

        if done: 
            with open(self._logdir, "wb") as in_file:
                pickle.dump(self._logdata, in_file)
        
        else: 
            obs = self.state
            # obs = np.expand_dims(obs, axis=0)
            combine = np.append(obs, action)
            if len(self._logdata) == 0 :
                self._logdata = combine 
            else: 
                self._logdata = np.vstack([self._logdata, combine])
