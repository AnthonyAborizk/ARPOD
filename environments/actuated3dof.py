'''
Autonomous Rendezvous, Proximity Operations, and Docking (ARPOD)
3D Spacecraft Docking Environment
Created by: Anthony Aborizk 
Citations/Inspiration: Kyle Dunlap, Kai Delsing, Kerianne Hobbs, R. Scott Erwin, Christopher Jewison
Description:
	A deputy spacecraft is trying to dock with and relocate a chief spacecraft in
    Hill's frame.
Nomenclature
    x, y, z        :  position in CWH frame
    vx, vy,vz      :  velocity in CWH frame
    fx, fy, fz     :  thruster force

    ARPOD       :  Autonomous Rendezvous, Proximity Operations, and Docking
    CWH         :  Clohessy-Wiltshire-Hill
    DOF         :  Degrees of Freedom
Observation (Deputy):
	Type: Box(4)
	Num 	Observation 		Min 	Max
	0		x position			-Inf 	+Inf
	1		y position			-Inf 	+Inf
	2		z position			-Inf 	+Inf
	3		x velocity			-Inf 	+Inf
	4		y velocity			-Inf 	+Inf
	5		z velocity			-Inf 	+Inf
Actions (Continuous):
	Type: Continuous Box(3,)
	[X direction, Y direction, Reaction Wheel Torque]
	Each value between -2 and +2 Newtons or -181.3 and +181.3 rad/s^2
Episode Termination:
	Deputy hits chief
	Deputy goes out of bounds
	Out of time
Integrators:
	RK45
''' 

import os
import time
import pickle
import numpy as np
import gymnasium as gym
from scipy import integrate
from gymnasium import spaces
from math import pi, sin, cos
from gymnasium.utils import seeding
from pyparsing import java_style_comment
from environments.rendering import DockingRender as render

gym.logger.set_level(40)

class ActuatedDocking3DOF(gym.Env):

    def __init__(self, logdir=None):
        '''
        #####################################################################################################
        Some of these initial conditions are taken from  "A Spacecraft Benchmark Problem for Hybrid Control 
        and Estimation" by Jewison and Erwin and others are taken from C. D. Petersen, S. Phillips, K. L. 
        Hobbs and L. Kendra, "CHALLENGE PROBLEM: ASSURED SATELLITE PROXIMITY OPERATIONS," in AAS/AIAA 
        Astrodynamics Specialist Conference, Big Sky, 2021. 
        #####################################################################################################
        '''

        #############################
        #   Environment Variables   #
        #############################
        self.position_deputy = 100      # m (Relative distance from chief)
        self.Md          = 12           # kg 
        self.n           = 0.001027     # rad/sec (mean motion) (circular orbit)
        self.TAU         = 1            # sec (time step)
        self.integrator  = 'RK45'       # Integrator type
        self.force_mag   = 2            # Newtons

        # m (In either direction)
        self.x_bound = 1.5 * self.position_deputy

        # m (In either direction)
        self.y_bound = 1.5 * self.position_deputy
        
        # m (In either direction)
        self.z_bound = 1.5 * self.position_deputy

        # m (|x| and |y| must be less than this to dock)
        self.pos_thresh = .1

        # m/s (Relative velocity must be less than this to dock)
        self.VEL_THRESH    = .2        # m/s
        self.max_time      = 4000*self.TAU   # seconds
        self.max_control   = 2500     # Newtons
        self.init_velocity = (self.position_deputy + 625) / 1125  # m/s (+/- x and y)

        #############################
        #   Rendering Variables     #
        #############################
        #! cannot render at this time due to gymnasiums 3D rendering issues
        
        #############################
        #     Logging Variables     #
        #############################        
        self._logdir  = logdir

        if logdir: 
            self.log(None, logdir, True)

        #############################
        #   OPENAI GYM Sim bounds   #
        #############################
        high = np.array([np.finfo(np.float32).max,  # x position (Max possible value +inf)
                         np.finfo(np.float32).max,  # y position
                         np.finfo(np.float32).max,  # orientation
                         2.998e8,                   # x velocity
                         2.998e8,                   # y velocity
                         2.998e8],                  # angular velocity
                        dtype=np.float32)
        low = -high

        self.action_space = spaces.Box(np.array([-self.force_mag, -self.force_mag, \
                                                 -self.force_mag]), np.array([self.force_mag,\
                                                  self.force_mag, self.force_mag]), dtype=np.float32)

        # Continuous observation space
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()  # Generate random seed


    def seed(self, seed=None):  
        '''
        #####################################################################################################
        Sets random seed
        #####################################################################################################
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self,*, seed=None, options=None):  
        '''
        #####################################################################################################
        This function resets the environment to its initial state, and returns the observation 
        of the environment corresponding to the initial state.
        
        Args:
            seed (int): seed for random number generator
            options (dict): dictionary of options for the environment
        Returns:
            observation (nparray): array of observations
            {} (dict): empty dictionary (for gymnasium compatibility)(can be replaced with info 
                       dictionsary)
        #####################################################################################################
        '''
        self.steps = -1         # Step counter
        self.control_input = 0  # Used to sum total control input for an episode

        theta_d = self.np_random.uniform(low=0, high=2*pi)

        self.x_deputy = self.position_deputy*cos(theta_d)  # m
        self.y_deputy = self.position_deputy*sin(theta_d)  # m
        self.z_deputy = self.np_random.uniform(low=0, high=self.position_deputy/2)  # m
        
        # Random x and y velocity
        x_dot = self.np_random.uniform(low=-self.init_velocity, high=self.init_velocity)  # m/s
        y_dot = self.np_random.uniform(low=-self.init_velocity, high=self.init_velocity)  # m/s
        z_dot = self.np_random.uniform(low=-self.init_velocity, high=self.init_velocity)  # m/s
        # (Relative distance from chief)
        self.rH = self.position_deputy  # m 

        # orientation of chaser
        self.state = np.array([self.x_deputy, self.y_deputy,self.z_deputy, x_dot, y_dot, z_dot])

        return self.state, {}


    def get_reward(self, obs, actions, obs_old, hstep):
        '''
        #####################################################################################################
        Calculates the reward for executing actions. 
        Args:
            observations (nparray): array of observations
            actions (nparray): array of actions
            hstep (int): mpc horizon step
        Returns:
            rewards this step and done conditions: rew and done

            This function is RL specific. Incomplete.
        #####################################################################################################
        '''

        x   = obs[0]  
        y   = obs[1]   
        z   = obs[2]

        # check dones conditions
        if (abs(y) <= self.pos_thresh) and abs(x) <= self.pos_thresh and abs(z) <= self.pos_thresh:
            done = True
        elif (abs(y) > self.y_bound) or (abs(x) > self.x_bound) or (abs(z) > self.z_bound):
            done = True
        elif (self.steps+hstep) * self.TAU >= self.max_time-1:
            done = True
        else:
            done = False

        return {}, done


    def step(self, action):
        '''
        #####################################################################################################
        This function takes an action as an input and applies it to the environment, which 
        leads to the environment transitioning to a new state.

        Args:
            action (nparray): array of actions
        Returns:
            observation (nparray): array of observations
            reward (float): reward for executing action
            truncated (bool): whether or not the episode is over
            done (bool): whether or not the episode is over
            {} (dict): empty dictionary (for gymnasium compatibility)(can be replaced with info dictionsary)
        #####################################################################################################
        '''
        self.steps += 1  # step counter

        # Clip action to be within boundaries
        action = np.clip(action, [-self.force_mag, -self.force_mag, -self.force_mag],\
                                 [self.force_mag, self.force_mag, self.force_mag])

        # Extract current state data
        x, y, z, x_dot, y_dot, z_dot = self.state
        self.x_force, self.y_force, self.z_force = action

        # Add total force for given time period
        self.control_input += (abs(self.x_force) + abs(self.y_force) + abs(self.z_force)) * self.TAU
        
        # Integrate Acceleration and Velocity
        if self.integrator == 'RK45':
            # Define acceleration functions
            def p_dot(t, state): 
                '''
                ################################################################################
                This function defines the acceleration of the spacecraft in the CWH frame. 

                x     = vx 
                y     = vy
                z     = vz
                vx    = 3n^2x + 2nvy + fx/M*W1
                vy    = -2nvx + fx/M*W2
                vz    = -fx/M*W3

                Args:
                    t (float): time
                    x (nparray): array of states
                Returns:
                    nparray: array of velocities
                ################################################################################
                '''
                x, y, z, vx, vy, vz = state

                x_acc = (3 * self.n ** 2 * x) + (2 * self.n * y_dot) + self.x_force / self.Md
                y_acc = (-2 * self.n * x_dot) + self.y_force / self.Md
                z_acc = -self.n**2*z + self.z_force/self.Md

                return np.array([vx, vy, vz, x_acc, y_acc, z_acc])
            
            x, y, z, vx, vy, vz = integrate.solve_ivp(p_dot, (0, self.TAU), self.state).y[:, -1] 
        
        observation = np.array([x, y, z, vx, vy, vz])        
       
        # Relative distance between deputy and chief (assume chief is always at origin)
        self.rH = np.sqrt(x**2 + y**2 + z**2)
        self.vH = np.sqrt(x_dot**2 + y_dot**2 + z_dot**2)  # Velocity Magnitude
        self.vH_max = 2 * self.n * self.rH + self.VEL_THRESH  # Max Velocity
        self.vH_min = 1/2 * self.n * self.rH - self.VEL_THRESH  # Min Velocity
        reward, done = self.get_reward(observation, action, self.state, 0)
        self.state = observation

        truncated = False
        if self.steps*self.TAU > self.max_time:
            truncated = True

        if self._logdir:
            self.log(action)

        if done and self._logdir: 
            self.log(action, done=done)

        return self.state, reward, done, truncated, {}

   
    def render(self, mode):
        render.renderSim(self, mode='human')


    def close(self):
        render.close(self)


    def log(self, action=None, data_path=None, initialize=False, done=False):
        '''
        logs observations also logs actions. '''
        if data_path is None:
            pass
        
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
