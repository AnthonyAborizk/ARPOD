'''
Nomenclature
    x, y, z       :  position in CWH frame
    vx, vy, vz    :  velocity in CWH frame
    mu1, mu2, mu3 :  MRP's
    w1, w2, w3    :  angular velocity in body frame
    nu1, nu2, nu3 :  wheel angular velocity in body frame
    u1, u2, u3    :  reaction wheel torque
    fx,fy,fz      :  thruster force
    zeta          :  regular MRP's
    sigma         :  shadow MRP's

    ARPOD         :  Autonomous Rendezvous, Proximity Operations, and Docking
    CWH           :  Clohessy-Wiltshire-Hill
    DOF           :  Degrees of Freedom
    MRP           :  Modified Rodrigues Parameters
    q             :  Quaternion

Summary
    This environment is a 6DOF fully-actuated ARPOD environment. 
    The translational dynamics are modeled in the CWH frame. 
    The rotational dynamics are modeled using MRP's with torques applied 
    via reaction wheels. The chaser's body fixed frame is aligned with the
    target's body fixed frame when all MRP's are zero.
    "under-actuated" means that the thruster force is not applied allong the 
    local hoizontal axis of the body fixed frame.   
'''

#! ###################################
#! THIS ENVIRONMENT IS NOT FINISHED !#
#! ###################################

import os
import time 
import pickle
import numpy as np
import gymnasium as gym
from scipy import integrate
from gymnasium import spaces
from environments.utils import *
from gymnasium.utils import seeding
from math import pi, sin, cos, sqrt
from pyparsing import java_style_comment
from environments.rendering import DockingRender as render

gym.logger.set_level(40) # show errors only. ignore warnings

class ActuatedDocking6DOF(gym.Env):

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
        self.Md = 12                    # kg 
        self.n   = 0.001027             # rad/sec (mean motion) (circular orbit)
        self.TAU = 1                    # sec (time step)
        self.integrator = 'RK45'        # Integration method
        self.force_mag = 2                 # Newtons
        self.u_mag = 181.3              # rad s^-2 max reaction wheel acceleration
        self.J1 = 0.022                 # kg-m**2 (1/2 M*R**2)
        self.J2 = 0.044                 # kg-m**2 (1/2 M*R**2)
        self.J3 = 0.056                 # kg-m**2 (1/2 M*R**2)
        self.Jw = 4.1e-5                # kg-m**2 reaction wheel spin axis mass moment of inertia

        # m (In either direction)
        self.x_bound = 2 * self.position_deputy

        # m (In either direction)
        self.y_bound = 2 * self.position_deputy
        
        #m (In either direction)
        self.z_bound = 2 * self.position_deputy

        # m (|x| and |y| and |z| must be less than this to dock)
        self.pos_thresh = .1

        # rad (Relative angle must be less than this to dock)
        self.theta_thresh = 5 * pi / 180   

        # rad/s (Relative angular velocity must be less than this to dock)
        self.theta_dot_thresh = 2 * pi / 180

        # m/s (Relative velocity must be less than this to dock)
        self.vel_thresh     = .2                # m/s
        self.max_time       = 4000*self.TAU     # max time allowed for episode
        self.max_control    = 2500              # Newtons
        self.init_velocity  = (self.position_deputy + 625) / 1125  # m/s (+/- x and y)
        self.init_theta_dot = pi / 180          # angular velocity within [-2, 2] deg/s

        #For Tensorboard Plots#
        self.success  = 0            # Used to count success rate for an epoch
        self.failure  = 0            # Used to count out of bounds rate for an epoch
        self.overtime = 0            # Used to count over max time/control for an epoch
        self.crash    = 0            # Used to count crash rate for an epoch

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
                         np.finfo(np.float32).max,  # z position
                         2.998e8,                   # x velocity (cannot exceed the speed of light)
                         2.998e8,                   # y velocity
                         2.998e8,                   # z velocity
                         1,                         # x angle     
                         1,                         # y angle
                         1,                         # z angle    
                         2.998e8,                   # x angular velocity
                         2.998e8,                   # y angular velocity
                         2.998e8,                   # z angular velocity
                         2.998e8,                   # x wheel velocity
                         2.998e8,                   # y wheel velocity
                         2.998e8],                  # z wheel velocity
                        dtype=np.float32)
        low = -high

        self.action_space = spaces.Box(np.array([-self.u_mag, -self.u_mag, -self.u_mag, self.force_mag]),\
                                       np.array([self.u_mag, self.u_mag, self.u_mag, self.force_mag]), dtype=np.float32)

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

        # called before each episode
        self.seed(seed)
        self.steps         = -1 # Step counter
        self.control_input = 0  # Used to sum total control input for an episode

        # randomly generate x, y, z values, normalize and scale to desired radial distance
        x   = self.np_random.uniform(low=-1, high=1) 
        y   = self.np_random.uniform(low=-1, high=1) 
        z   = self.np_random.uniform(low=-1, high=1) 
        mag = sqrt(x**2 + y**2 + z**2)
        x   = x/mag*self.position_deputy # m
        y   = y/mag*self.position_deputy # m
        z   = z/mag*self.position_deputy # m

        self.rH = self.position_deputy   # m 

        # randomly generate velocity (vx, vy, vz) values
        vx  = self.np_random.uniform(low=-self.init_velocity, high=self.init_velocity)  # m/s
        vy  = self.np_random.uniform(low=-self.init_velocity, high=self.init_velocity)  # m/s
        vz  = self.np_random.uniform(low=-self.init_velocity, high=self.init_velocity)  # m/s

        # randomly generate MRP and angular velocity values
        ## initial Euler angles (rad)
        Gamma = 2*pi*self.np_random.uniform(low=0, high=1, size=[3])
        # convert to MRP
        self.MRP   = EulerConvert(Gamma) # rad
        # separate shadow MRP's and regular MRP's
        zeta  = self.MRP[0:3]
        sigma = self.MRP[3:]

        zeta_norm  = np.linalg.norm(zeta)
        sigma_norm = np.linalg.norm(sigma)

        if zeta_norm <= 1:
            # MRP is regular
            mu = zeta   # rad
        elif sigma_norm < 1:
            # MRP is shadow
            mu = sigma   # rad
        else: 
            ValueError('MRP is not valid')

        # randomly generate angular velocity values
        omega = (-np.ones([3,]) + 2*self.np_random.uniform(low=0, high=1, size=(3,)))
        # randomly generate wheel angular velocity values
        nu = self.np_random.uniform(low=-1, high=1, size=[3])  # rad/s
        
        trans_states = np.array([x, y, z, vx, vy, vz]) # translation states
        self.state = np.hstack([trans_states, mu, omega, nu])

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
        action = np.clip(action, [-self.u_mag, -self.u_mag, -self.u_mag, -self.force_mag, -self.force_mag, -self.force_mag],\
                                 [self.u_mag, self.u_mag, self.u_mag, self.force_mag, self.force_mag, self.force_mag])

        # Extract current state and action data
        x, y, z, vx, vy, vz, mu1, mu2, mu3, w1, w2, w3, nu1, nu2, nu3 = self.state
        self.u1, self.u2, self.u3, self.fx, self.fy, self.fz = action

        mu = np.sqrt(mu1**2 + mu2**2 + mu3**2)

        # Integrate Acceleration and Velocity
        if self.integrator == 'RK45':  # Runge-Kutta Integrator
            # Define acceleration functions

            def p_dot(t, state):
                '''
                ################################################################################
                This function defines the acceleration of the spacecraft in the CWH frame. 
                The derivation of MRP's and angular velocity are elaborated on in the "full
                dynamics" file
                x     = x_dot 
                y     = y_dot
                z     = z_dot
                vx    = 3n^2x + 2nvy + fx/M*W1
                vy    = -2nvx + fx/M*W2
                vz    = -n^2z + fx/M*W3
                        | zeta_dot,  if |zeta| <= 1 
                mu    = |
                        | sigma_dot, if |sigma| < 1
                omega = omega_dot
                nu    = control input
                
                Args:
                    t (float): time
                    x (nparray): array of states
                Returns:
                    nparray: array of velocities
                ################################################################################
                '''
                # Extract current state data
                x     = state[0]
                # y     = state[1] # not used
                z     = state[2]
                vx    = state[3]
                vy    = state[4]
                vz    = state[5] # not used
                zeta  = state[6:9].reshape(3,1)
                sigma = state[9:12].reshape(3,1)
                omega = state[12:15].reshape(3,1)
                nu    = state[15:18] # not used
                
                # check if MRP is regular or shadow
                if np.linalg.norm(zeta) <= 1:
                    mu = zeta
                else:
                    mu = sigma

                # determine how translation is coupled to rotation
                mag = np.linalg.norm(mu)
                W1 = 4 / ( mag**2 + 1 )**2 * (-2*(mu[1,0]**2 + mu[2,0]**2) + (mag**2+1)**2/4)
                W2 = 4 / ( mag**2 + 1 )**2 * (2*mu[0,0]*mu[1,0] + mu[2,0]*(mag**2-1))
                W3 = 4 / ( mag**2 + 1 )**2 * (2*mu[0,0]*mu[2,0] - mu[1,0]*(mag**2-1))

                # calculate acceleration
                x_acl = (3 * self.n ** 2 * x) + (2 * self.n * vy) + (self.fx/ self.Md)*W1
                y_acl = (-2 * self.n * vx) + (self.fx / self.Md)*W2
                z_acl = (-self.n**2*z) + (self.fx/self.Md)*W3

                trans_states = np.array([vx, vy, vz, x_acl, y_acl, z_acl], dtype=np.float32)

                # calculate change in MRPs, shadow and regular
                dzetadt  = -1/2*(1/2*(1 - zeta.T@zeta)*np.eye(3) - skew(zeta) + zeta@zeta.T)@omega
                dsigmadt = -1/2*(1/2*(1 - sigma.T@sigma)*np.eye(3) - skew(sigma) + sigma@sigma.T)@omega
                
                dMRPdt = np.array(np.vstack([dzetadt, dsigmadt]).squeeze(), dtype=np.float32)
                
                # calculate change in angular velocity
                w1, w2, w3 = omega
                a = 4*self.n / ( mag**2 + 1 )**2 * (2*mu[1]*mu[2] - mu[0]*(mag**2-1))
                b = 4*self.n / ( mag**2 + 1 )**2 * (mag**2+1)**2/4 -2*(mu[0]**2 + mu[1]**2)
                c = 4*self.n / ( mag**2 + 1 )**2 * (2*mu[0]*mu[2] + mu[1]*(mag**2-1))

                w1_dot = w2*b - w3*a + self.Jw/self.J1*(nu[1]*(w3+b) - nu[2]*(w2+a)-self.u1)\
                      + 1/self.J1*(self.J2 - self.J3)*(w2+a)*(w3+b)
                
                w2_dot = w3*c - w1*b + self.Jw/self.J2*(nu[2]*(w1+c) - nu[0]*(w3+b)-self.u2)\
                      + 1/self.J2*(self.J3 - self.J1)*(w3+b)*(w1+c)
                
                w3_dot = w1*a - w2*c + self.Jw/self.J3*(nu[0]*(w2+a) - nu[1]*(w1+c)-self.u3)\
                      + 1/self.J3*(self.J1 - self.J2)*(w1+c)*(w2+a)
                
                omega_dot = np.array([w1_dot, w2_dot, w3_dot], dtype=np.float32).squeeze()

                # calculate change in wheel angular velocity
                nu_dot = np.array([self.u1, self.u2, self.u3])

                return np.hstack([trans_states, dMRPdt, omega_dot, nu_dot])

            # Integrate acceleration to calculate velocity
            trans_states = np.array([x, y, z, vx, vy, vz])
            omega        = np.array([w1, w2, w3])
            nu           = np.array([nu1, nu2, nu3])
            inits        = np.hstack([trans_states, self.MRP, omega, nu])
            outputs      = integrate.solve_ivp(p_dot, (0, self.TAU), inits).y[:, -1] 

            # Define new observation state
            zeta  = outputs[6:9]
            sigma = outputs[9:12]
            
            # check if MRP is regular or shadow
            if np.linalg.norm(zeta) <= 1:
                mu = zeta
            else:
                mu = sigma

            obs = np.hstack([outputs[0:6], mu, outputs[12:18]])

        # Relative distance between deputy and chief (assume chief is always at origin)
        self.rH     = np.sqrt(obs[0]**2 + obs[1]**2 + obs[2]**2)
        self.vH     = np.sqrt(obs[3]**2 + obs[4]**2 + obs[5]**2)  # Velocity Magnitude
        self.vH_max = 2 * self.n * self.rH + self.vel_thresh    # Max Velocity
        self.vH_min = 1/2 * self.n * self.rH - self.vel_thresh  # Min Velocity
        reward, done = self.get_reward(obs, action, self.state, 0)
        self.state  = obs

        truncated = False
        if self.steps*self.TAU > self.max_time:
            truncated = True

        if self._logdir:
            self.log(action)

        if done and self._logdir: 
            self.log(action, done=done)

        return self.state, reward, done, truncated, {}
    

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

