'''
2D Spacecraft Docking Environment

Created by Kyle Dunlap and Kai Delsing
Mentor: Kerianne Hobbs

Modified by Anthony Aborizk
Mentor: Scott Nivison

Modified by Josh Thompson
Mentor: Anthony Aborizk

Description:
	A deputy (chaser) spacecraft is trying to dock with the chief (target) spacecraft in Hill's frame. 
	This is the first sub-policy in a hierarchical deep reinforcement learning model. 
 
 dynamics

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
	+1 for reaching the chief faster than the previous model
	-1 for going out of bounds
	-1 for running out of time

Starting State:
	Deputy start 10000 m away from chief at random angle
	x and y velocity are both between -1.44 and +1.44 m/s

Episode Termination:
	Deputy docks with chief
	Deputy hits chief
	Deputy goes out of bounds
	Out of time

Integrators:
	Euler: Simplest and quickest
	Quad: ~4 times slower than Euler
	RK45: ~10 times slower than Quad
'''

import gym
from gym import spaces
from gym.utils import seeding
import math
import numpy as np
from pyparsing import java_style_comment
from scipy import integrate
from envs.docking.rendering import DockingRender as render
import random


class SpacecraftDocking(gym.Env):

    def __init__(self):
        
        #define position of target as origin
        self.x_chief = 0  # m
        self.y_chief = 0  # m
        
        #starting distance of chaser
        self.position_deputy = 10000  # m (Relative distance from chief)
        
        self.mass_deputy = 12       # kg mass of chaser
        self.n = 0.001027           # rad/sec (mean motion)
        self.tau = 1                # sec (time step) - time constant; how fast chaser is moving
        
        # Either 'Quad', 'RK45', or 'Euler' (default)
        self.integrator = 'Euler' #define which integrator will be used
        self.force_magnitude = 1    # Newton(s)
        
        # m (In either direction)
        self.x_threshold = 1.5 * self.position_deputy
        
        # m (In either direction)
        self.y_threshold = 1.5 * self.position_deputy
        
        # m (|x| and |y| must be less than this to dock)
        self.pos_threshold = 0.1
        
        # m/s (Relative velocity must be less than this to dock)
        self.vel_threshold = .2
        
        self.max_time = 4000        # seconds
        self.max_control = 2500     # Newtons
        self.init_velocity = (self.position_deputy + 625) / \
            1125  # m/s (+/- x and y)
        # Changes reward for different RTA, either 'NoRTA', 'CBF', 'SVL', 'ASIF', or 'SBSF'
        self.RTA_reward = 'NoRTA'

        #For Tensorboard Plots
        self.RTA_on = False  # Flag for if RTA is on or not, used for rewards
        self.success = 0  # Used to count success rate for an epoch
        self.failure = 0  # Used to count out of bounds rate for an epoch
        self.overtime = 0  # Used to count over max time/control for an epoch
        self.crash = 0     # Used to count crash rate for an epoch

        #Thrust & Particle Variables
        # what type of thrust visualization to use. 'Particle', 'Block', 'None'
        self.thrustVis = 'Particle'
        self.particles = []         # list containing particle references
        self.p_obj = []             # list containing particle objects
        self.trans = []             # list containing particle
        self.p_velocity = 20        # velocity of force particle
        self.p_ttl = 4              # (steps) time to live per particle
        # (deg) the variation of launch angle (multiply by 2 to get full angle)
        self.p_var = 3

        #Ellipse Variables
        #define dimensions of circle (boundries of each phase)
        self.ellipse_a1 = 1000      # m
        self.ellipse_a2 = 100       # m
        self.ellipse_a3=10000       #m
        self.ellipse_quality = 150  # 1/x * pi
        
        #Trace Variables
        self.trace = 8       # (steps) spacing between trace dots
        self.traceMin = True  # sets trace size to 1 (minimum) if true
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
        self.velocityArrow = True
        self.forceArrow = True                  # if force arrow is shown
        self.bg_color = (0, 0, .15)             # r,g,b
        #color of background (sky)
        
        self.stars = 400             # sets number of stars; adding more makes program run slower
        # Set to true to print termination condition
        self.termination_condition = False
        
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

      #  self.reset()  # Reset environment when initialized

    def action_select(self):  # Defines action type
        self.action_type = 'Discrete'

    def seed(self, seed=None):  # Sets random seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):  # called before each episode; reset simulation
        self.steps = -1  # Step counter
        self.control_input = 0  # Used to sum total control input for an episode

        # Use random angle to calculate x and y position
        # random angle, starts 10km away
        theta = self.np_random.uniform(low=0, high=2*math.pi) #rad
        #gives original angle relative to target; controls start angle
       
        self.x_deputy = self.position_deputy*math.cos(theta)  # m (x distance to chaser)
        self.y_deputy = self.position_deputy*math.sin(theta)  # m (y distance to chaser)   
            
        # #compare to origin (target)
        # print('x is',self.x_deputy,'m')
        # print('y is',self.y_deputy,'m')  
        # print('angle relative to target is',theta*(180/math.pi),'degrees') 
        # if self.x_deputy<self.x_chief:
        #     print('the chaser needs to move right')     
        # elif self.x_deputy>self.x_chief:
        #     print('the chaser needs to move left')
            
        # if self.y_deputy<self.y_chief:
        #     print('the chaser needs to move up')     
        # elif self.y_deputy>self.y_chief:
        #     print('the chaser needs to move down')

        # Random x and y velocity
        #random original velocity
        #controls direction chaser originally goes
        x_dot = self.np_random.uniform(
            low=-self.init_velocity, high=self.init_velocity)  # m/s
        y_dot = self.np_random.uniform(
            low=-self.init_velocity, high=self.init_velocity)  # m/s
        # print('xdot is',x_dot,'m/s')
        # print('ydot is',y_dot,'m/s')
        
        self.rH = self.position_deputy  # m (Relative distance from chief)

        # Define observation state
        #state vector array
        self.state = np.array([self.x_deputy, self.y_deputy, x_dot, y_dot])
        return self.state
    

    def mpc_reward(self, observations, actions, steps):
        '''
        observations : input state space
        actions : selected actions
        steps : counter for step of path in simulation
        
        '''
        if (len(observations.shape) == 1):
            observations = np.expand_dims(observations, axis=0)
            actions = np.expand_dims(actions, axis=0)

        xx = observations[:, 0]
        yy = observations[:, 1]  
        x_dot_obs = observations[:, 2]
        y_dot_obs = observations[:, 3]
        rew = []
        new_state = [[]]
        for i in range(xx.shape[0]):
            rH_old = np.sqrt(xx[i]**2 + yy[i]**2)

            if actions.shape[0] ==1:
                actions = actions.reshape(actions.shape[1], 2)

            x_force = actions[i, 0]
            y_force = actions[i, 1]

            # control_input += (abs(x_force) +
            #                        abs(y_force)) * self.tau
            
            # Integrate acceleration to calculate velocity
            x_acc = (3 * self.n ** 2 * xx[i]) + (2 * self.n * y_dot_obs[i]) + \
                (x_force / self.mass_deputy)
            y_acc = (-2 * self.n * x_dot_obs[i]) + \
                (y_force / self.mass_deputy)
                
            # Integrate acceleration to calculate velocity
            x_dot = x_dot_obs[i] + x_acc * self.tau
            y_dot = y_dot_obs[i] + y_acc * self.tau
            
            # Integrate velocity to calculate position
            x = xx[i] + x_dot * self.tau
            y = yy[i] + y_dot * self.tau
            
            new_state += [x,y,x_dot,y_dot]
            # Relative distance between deputy and chief (assume chief is always at origin)
            rH = np.sqrt(x**2 + y**2)
            #new distance
            
            vH = np.sqrt(x_dot**2 + y_dot**2)  # Velocity Magnitude
            #new velocity
            
            vH_max = 2 * self.n * rH + self.vel_threshold  # Max Velocity
            vH_min = 1/2 * self.n * rH - self.vel_threshold  # Min Velocity

            done = bool(
                (abs(x) <= self.pos_threshold and abs(y) <= self.pos_threshold)
                or abs(x) > self.x_threshold
                or abs(y) > self.y_threshold
                # or self.control_input > self.max_control
                or steps * self.tau > self.max_time
            )
            rH = np.sqrt(x**2 + y**2)

            if not done:
                reward = (-1+rH_old-rH)/2000 * self.tau
                if rH < rH_old:
                    reward += 1
                else: 
                    reward -= 1
            #     if vH < vH_min:
            #         # Negative reward for being below min velocity
            #         reward += -0.0075*abs(vH-vH_min) * self.tau
            #        # print('vH<vH_min')

            #     if self.RTA_on or vH > vH_max:
            #         if self.RTA_reward == 'NoRTA':
            #             # Negative reward for being over max velocity
            #             reward += -0.0035*abs(vH-vH_max) * self.tau
            #          #   print('over max velocity')
            #         else:
            #             reward += -0.001 * self.tau  # Negative reward if RTA is on
            #           #  print('rta is on')

            #     if vH < 2*self.vel_threshold and (self.RTA_on or vH < vH_min or vH > vH_max):
            #         if self.RTA_reward == 'NoRTA':
            #             # Larger negative reward if violating constraint close to docking
            #             reward += -0.0075/2 * self.tau
            #            # print('violate close to docking')
            #         else:
            #             # Larger negative reward if violating constraint close to docking
            #             reward += -0.005/2 * self.tau
            #           #  print('violate close to docking too')

            # elif abs(x) <= self.pos_threshold and abs(y) <= self.pos_threshold:
            #     if vH > self.vel_threshold:
            #         reward = -0.001  # Negative reward for crashing
            #         self.crash += 1  # Track crash
            #     #    print('crash')
            #     else:
            #         reward = 1  # +1 for docking
            #         self.success += 1  # Track success
            #       #  print("SUCCESS!!!!")
            # elif steps * self.tau > self.max_time:  # self.control_input > self.max_control or
            #     reward = -1  # -1 for over max time or control
            #     self.overtime += 1  # Track overtime
            #    # print('over max time')

            else:
                reward = -1        # -1 for going out of bounds
                self.failure += 1  # Track failure
             #   print('track failure')
            rew.append(reward)
        new_state=np.array(new_state[1:])

        return rew, new_state,done

    def get_reward(self, observations, actions):
        if (len(observations.shape) == 1):
            observations = np.expand_dims(observations, axis=0)
            actions = np.expand_dims(actions, axis=0)

        xx = observations[:, 0]
        yy = observations[:, 1]  # Observations
        rH_old = np.sqrt(self.state[0]**2 + self.state[1]**2)
        rew = []
        for i in range(xx.shape[0]):
            x = xx[i]
            y = yy[i]

            done = bool(
                (abs(x) <= self.pos_threshold and abs(y) <= self.pos_threshold)
                or abs(x) > self.x_threshold
                or abs(y) > self.y_threshold
                or self.control_input > self.max_control
                or self.steps * self.tau > self.max_time
            )
            rH = np.sqrt(x**2 + y**2)

            if not done:
                reward = (-1+rH_old-rH)/2000 * self.tau

                if self.vH < self.vH_min:
                    # Negative reward for being below min velocity
                    reward += -0.0075*abs(self.vH-self.vH_min) * self.tau
                 #   print('below min velocity')

                if self.RTA_on or self.vH > self.vH_max:
                    if self.RTA_reward == 'NoRTA':
                        # Negative reward for being over max velocity
                        reward += -0.0035*abs(self.vH-self.vH_max) * self.tau
                     #   print('over max velocity')
                    else:
                        reward += -0.001 * self.tau  # Negative reward if RTA is on
                      #  print('rta on')

                if self.vH < 2*self.vel_threshold and (self.RTA_on or self.vH < self.vH_min or self.vH > self.vH_max):
                    if self.RTA_reward == 'NoRTA':
                        # Larger negative reward if violating constraint close to docking
                        reward += -0.0075/2 * self.tau
                      #  print('close to docking')
                    else:
                        # Larger negative reward if violating constraint close to docking
                        reward += -0.005/2 * self.tau
                      #  print('close to docking too')

            elif abs(x) <= self.pos_threshold and abs(y) <= self.pos_threshold:
                if self.vH > self.vel_threshold:
                    reward = -0.001  # Negative reward for crashing
                    self.crash += 1  # Track crash
                   # print('crash')
                else:
                    reward = 1  # +1 for docking
                    self.success += 1  # Track success
                 #   print("SUCCESS!!!!")
            elif self.control_input > self.max_control or self.steps * self.tau > self.max_time:
                reward = -1  # -1 for over max time or control
                self.overtime += 1  # Track overtime
             #   print('over max time')

            else:
                reward = -1  # -1 for going out of bounds
                self.failure += 1  # Track failure
               # print('track failure')
            rew.append(reward)

        return rew, done

        # return self.reward_dict['r_total'], done

    def step(self, action):
        self.steps += 1  # step counter

        if self.action_type == 'Discrete':
            # Stop program if invalid action is used
            assert self.action_space.contains(action), "Invalid action"
        else:
            # Clip action to be within boundaries - only for continuous
            #makes actions that are outside force_magnitude equal 
            #to plus or minus force_magnitude
            action = np.clip(action, -self.force_magnitude,
                             self.force_magnitude)
         

        # Extract current state data
        x, y, x_dot, y_dot = self.state
        current_state=self.state #current state

        # if self.action_type == 'Discrete':  # Discrete action space
        #     if action == 0:
        #         self.x_force = -self.force_magnitude
        #         self.y_force = -self.force_magnitude
        #     elif action == 1:
        #         self.x_force = -self.force_magnitude
        #         self.y_force = 0
        #     elif action == 2:
        #         self.x_force = -self.force_magnitude
        #         self.y_force = self.force_magnitude
        #     elif action == 3:
        #         self.x_force = 0
        #         self.y_force = -self.force_magnitude
        #     elif action == 4:
        #         self.x_force = 0
        #         self.y_force = 0
        #     elif action == 5:
        #         self.x_force = 0
        #         self.y_force = self.force_magnitude
        #     elif action == 6:
        #         self.x_force = self.force_magnitude
        #         self.y_force = -self.force_magnitude
        #     elif action == 7:
        #         self.x_force = self.force_magnitude
        #         self.y_force = 0
        #     elif action == 8:
        #         self.x_force = self.force_magnitude
        #         self.y_force = self.force_magnitude

        # else:  # Continuous action space
        action=action.tolist()
       
        self.x_force, self.y_force = action

        # Add total force for given time period
        self.control_input += (abs(self.x_force) +
                               abs(self.y_force)) * self.tau

        # Integrate Acceleration and Velocity
        if self.integrator == 'RK45':  # Runge-Kutta Integrator
            # Define acceleration functions
            def x_acc_int(t, x):
                return (3 * self.n ** 2 * x) + (2 * self.n * y_dot) + (self.x_force / self.mass_deputy)

            def y_acc_int(t, y):
                return (-2 * self.n * x_dot) + (self.y_force / self.mass_deputy)
            # Integrate acceleration to calculate velocity
            x_dot = integrate.solve_ivp(
                x_acc_int, (0, self.tau), [x_dot]).y[-1, -1]
            y_dot = integrate.solve_ivp(
                y_acc_int, (0, self.tau), [y_dot]).y[-1, -1]
            # Define velocity functions

            def vel_int(t, x):
                return x_dot, y_dot
            # Integrate velocity to calculate position
            xtemp, ytemp = integrate.solve_ivp(
                vel_int, (0, self.tau), [x, y]).y
            x = xtemp[-1]
            y = ytemp[-1]

        elif self.integrator == 'Euler':  # Simple Euler Integrator
            # Define acceleration functions
            #CW Equations
            x_acc = (3 * self.n ** 2 * x) + (2 * self.n * y_dot) + \
                (self.x_force / self.mass_deputy)
            y_acc = (-2 * self.n * x_dot) + (self.y_force / self.mass_deputy)
            # Integrate acceleration to calculate velocity
            x_dot = x_dot + x_acc * self.tau
            y_dot = y_dot + y_acc * self.tau
            # Integrate velocity to calculate position
            x = x + x_dot * self.tau
            y = y + y_dot * self.tau
            #gives new position
      

        else:  # Default 'Quad' Integrator
            # Integrate acceleration to calculate velocity
            x_dot = x_dot + integrate.quad(lambda x: (3 * self.n ** 2 * x) + (
                2 * self.n * y_dot) + (self.x_force / self.mass_deputy), 0, self.tau)[0]
            y_dot = y_dot + integrate.quad(lambda y: (-2 * self.n * x_dot) + (
                self.y_force / self.mass_deputy), 0, self.tau)[0]
            # Integrate velocity to calculate position
            x = x + integrate.quad(lambda x: x_dot, 0, self.tau)[0]
            y = y + integrate.quad(lambda y: y_dot, 0, self.tau)[0]

        # Define new observation state
        observation = np.array([x, y, x_dot, y_dot]) 
        #new state from CW Equations

        # Relative distance between deputy and chief (assume chief is always at origin)
        self.rH = np.sqrt(x**2 + y**2) #distance magnitude
        self.vH = np.sqrt(x_dot**2 + y_dot**2)  # Velocity Magnitude
        self.vH_max = 2 * self.n * self.rH + self.vel_threshold  # Max Velocity
        self.vH_min = 1/2 * self.n * self.rH - self.vel_threshold  # Min Velocity
        rew, done = self.get_reward(observation, action)
        self.state = observation
    
        new_state=self.state
        #new state
         
        #reward = {}
        # reward['rew'] = rew
        # reward['crash'] = self.crash
        # reward['failure'] = self.failure
        # reward['overtime'] = self.overtime
        # reward['success'] = self.success
        
        
                

        #conditions for phases    
        if self.rH>10000:
            #phase 1, 2 DOF
            self.state=self.state #dynamics is the same
            self.alpha=math.atan(self.state[2]/self.state[1])
            self.y_meas=self.alpha+self.noise
        elif self.rH<10000 and self.rH>1000:
            #phase 2, 2 DOF
            self.state=self.state #dynamics is the same
            self.alpha=math.atan(self.state[2]/self.state[1])
            self.y_meas=[self.alpha+self.noise,self.rH+self.noise]
        #elif self.rH<1000:
            #phase 3, 2 DOF
            #make cone for LOS
            
        
        
        
        
        #center of mass
        #self.x_com=sum()
        
        
        
        
        # #low force
        # #chemical and electric propulsion
        # self.force_chem=5
        # self.force_elec=5
        
    
        
        
        return current_state,new_state, rew, done, {}        
   
        # Used to check if velocity is over max velocity constraint
        
    def pred(self,act):
        n=0.001027   
        mass_deputy=12
        tau=1
        x, y, x_dot, y_dot = self.state
        x_force, y_force = act
        control_input=0
        # Add total force for given time period
        control_input += (abs(x_force) +
                               abs(y_force)) * tau
        # Define acceleration functions
        #CW Equations          
        #collect x and y force
        #initiate constants
      
        x_acc = (3 * n ** 2 * x) + (2 * n * y_dot) + \
                (x_force / mass_deputy)
        y_acc = (-2 * n * x_dot) + (y_force / mass_deputy)
        # Integrate acceleration to calculate velocity
        x_dot = x_dot + x_acc * tau
        y_dot = y_dot + y_acc * tau
        # Integrate velocity to calculate position
        x = x + x_dot * tau
        y = y + y_dot * tau
        #gives predicted position
        pred_state=x,y,x_dot,y_dot
        
        return pred_state    
    
    def check_velocity(self, x_force, y_force):
        # Extract current state data
        x, y, x_dot, y_dot = self.state
        # Define acceleration functions
        x_acc = (3 * self.n ** 2 * x) + (2 * self.n * y_dot) + \
            (x_force / self.mass_deputy)
        y_acc = (-2 * self.n * x_dot) + (y_force / self.mass_deputy)
        # Integrate acceleration to calculate velocity
        x_dot = x_dot + x_acc * self.tau
        y_dot = y_dot + y_acc * self.tau

        # Check max velocity, and return True if it is violating constraint
        rH = np.sqrt(x**2 + y**2)          # m, distance between deputy and chief
        vH = np.sqrt(x_dot**2 + y_dot**2)  # Velocity Magnitude
        vH_max = 2 * self.n * self.rH + self.vel_threshold  # Max Velocity

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

    # Run Time Assurance for discrete actions- based on velocity constraint
    def RTA(self, action):
        # Extract current state data
        x, y, x_dot, y_dot = self.state

        # Define force value for each possible action
        if action == 0:
            x_force = -self.force_magnitude
            y_force = -self.force_magnitude
        elif action == 1:
            x_force = -self.force_magnitude
            y_force = 0
        elif action == 2:
            x_force = -self.force_magnitude
            y_force = self.force_magnitude
        elif action == 3:
            x_force = 0
            y_force = -self.force_magnitude
        elif action == 4:
            x_force = 0
            y_force = 0
        elif action == 5:
            x_force = 0
            y_force = self.force_magnitude
        elif action == 6:
            x_force = self.force_magnitude
            y_force = -self.force_magnitude
        elif action == 7:
            x_force = self.force_magnitude
            y_force = 0
        elif action == 8:
            x_force = self.force_magnitude
            y_force = self.force_magnitude

        # Check if over max velocity constraint
        over_max_vel, _, _ = self.check_velocity(x_force, y_force)

        # If violating:
        if over_max_vel:
            # act_list is a list of possible actions that do not violate max velocity constraint
            action = []
            # Test all 9 actions (except for one already tested)
            for i in range(9):
                if i == action:
                    continue
                if i == 0:
                    x_force = -self.force_magnitude
                    y_force = -self.force_magnitude
                elif i == 1:
                    x_force = -self.force_magnitude
                    y_force = 0
                elif i == 2:
                    x_force = -self.force_magnitude
                    y_force = self.force_magnitude
                elif i == 3:
                    x_force = 0
                    y_force = -self.force_magnitude
                elif i == 4:
                    x_force = 0
                    y_force = 0
                elif i == 5:
                    x_force = 0
                    y_force = self.force_magnitude
                elif i == 6:
                    x_force = self.force_magnitude
                    y_force = -self.force_magnitude
                elif i == 7:
                    x_force = self.force_magnitude
                    y_force = 0
                elif i == 8:
                    x_force = self.force_magnitude
                    y_force = self.force_magnitude

                # Check if each is over max velocity
                over_max_vel, _, _ = self.check_velocity(x_force, y_force)

                # If that action does not violate max velocity constraint, append it to lists
                if not over_max_vel:
                    action.append(i)

            # Set RTA flag to True
            self.RTA_on = True

        # If it is not violating constraint
        else:
            self.RTA_on = False

        # If RTA is on, returns list of possible actions. If RTA is off, returns original action
        return action

    # Rendering Functions
    def render(self, mode):
        render.renderSim(self, mode='human')

    def close(self):
        render.close(self)

# Used to define 'spacecraft-docking-continuous-v0'


class SpacecraftDockingContinuous(SpacecraftDocking):
    def action_select(self):  # Defines continuous action space
        self.action_type = 'Continuous'
