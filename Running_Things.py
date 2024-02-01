# Running ARPOD Simulations in Python # 
# Author: Anthony Aborizk
# Date: 8/7/2023

# Note: Please follow the instructions in the README.md file before running this or any script

# The purpose of this script is to help the user familiarize themselves with the ARPOD 
# simulations included in this repository. The user can decide which version of the 
# ARPOD promblem they would like to run and a dummy controller will be used to generate
# the necessary control inputs to run the simulation. 

# The user can decide between the following simulations options: 
# 1. Fully-actuated ARPOD with a 3-DOF planar model
#    a. This included translation about the local vertical and local horizontal axes as
#       well as the angular momentum vector
#
# 2. Fully-actuated ARPOD with a 6-DOF model
#    a. This includes translation and rotation about all three axes
#
# 3. Under-actuated ARPOD with a 3-DOF planar model
#    a. This includes translation about the local vertical and local horizontal axes as
#       well as rotation about the angular momentum vector
#    b. This simulation assumes that the ARPOD is under-actuated (i.e. the chaser can only 
#       control a flywheel that is aligned with the angular momentum vector and a thruster
#       that is aligned with the local vertical axis)
#
# 4. Under-actuated ARPOD with a 6-DOF model
#    a. This includes translation and rotation about all three axes
#    b. This simulation assumes that the ARPOD is under-actuated (i.e. the chaser can
#       control all flywheels and a single thruster that is aligned with the local 
#       vertical axis)

# import necessary libraries
import numpy as np
from environments.utils import *
from environments.actuated3dof import ActuatedDocking3DOF
from environments.actuated6dof import ActuatedDocking6DOF
from environments.underactuated6dof import UnderactuatedDocking6DOF

#TODO Choose which environment to run: 
# - 'ActuatedDocking3DOF' for a fully-actuated 3-DOF model w/o rotation 
# - 'ActuatedDocking6DOF' for a fully-actuated 6-DOF model
# - 'UnderactuatedDocking6DOF' for an under-actuated 6-DOF model

env_name = 'ActuatedDocking3DOF'


if env_name == 'ActuatedDocking3DOF':
    env = ActuatedDocking3DOF()
elif env_name == 'ActuatedDocking6DOF':
    env = ActuatedDocking6DOF()
elif env_name == 'UnderactuatedDocking6DOF':
    env = UnderactuatedDocking6DOF()

# instantiate vairalbles
states, actions = [], []
done  = False
steps = 0

# collect the initial conditions
state, _ = env.reset()   

while not done:                     
    steps+=1
    states.append([state]) # collect states

    # Replace this whatever controller you want to use
    control_input = np.zeros(env.action_space.shape)
    actions.append(control_input) # collect actions

    # plug actions into agent, collect next states        
    state, reward, done, _, info = env.step(control_input) 
print('Done')
print(env.steps)
print('state: ', state)

# convert to numpy arrays
states  = np.array(states)
actions = np.array(actions)

plots(states, actions, env_name)