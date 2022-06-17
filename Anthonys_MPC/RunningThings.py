from hashlib import new
from envs.docking import SpacecraftDockingContinuous
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
startTime = time.time()


def policy(Horizon, num_seq, observation, env, steps):
    # num_seq is how many different randoms paths are to be taken
    # Horizon is how far to look into future in each path
    low = env.action_space.low
    high = env.action_space.high
    action_mat=np.random.rand(Horizon, num_seq, 2) * (high - low) + low
    #random Horizon x num_seq x 2 3D matrix of actions
    # build model predictive controller (MPC)
    all_rew_sum = []
    reward = np.zeros(num_seq)
    new_state=np.tile(observation, (num_seq,1)) #collect current state from original observation
    for j in range(Horizon):
        act=action_mat[j,:,:].reshape(num_seq, 2) #Horizon x 2
        rew,new_state,_=env.mpc_reward(new_state,act,steps) #collect returns from mpc_reward
        reward+=rew      #sum reward for each different path
    best_index = reward.argmax()
    chosen_action=action_mat[0,best_index,:] #collect first action of path that has highest reward

        
    return chosen_action


plt.figure()                        # Generate figure window
env = SpacecraftDockingContinuous() # Call environment script
current_state = env.reset()         # instantiate the envirionment, i.e. collect the initial conditions

for k in range(1000):                # sets time of simulation in real life
    steps=k
    chosen_action = policy(10,1000,current_state,env,steps)     # gives action with max reward (return of policy)
    current_state,new_state, _, _ = env.step(chosen_action) # plug actions into agent, collect next states

    if k % 25 == 0:     # step sizes are small so we are showing every 8th step
        #Can change this number to make things run faster/slower
        #controls size of step in simulation
        env.render(mode='human')

env.close() #to fix "python is likely shutting down" problem
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))



# #3D plot

# fig = plt.figure(figsize=(16, 16))  # Generate blank figure of size 16x16
# ax = fig.add_subplot(111, projection='3d')  # Project into 3 dims
# cmap = plt.cm.seismic   # define the color map you want to use. You can google different color maps if you want

# scat_plot = ax.scatter(rew, Horizon, num_seq, cmap=cmap)
# # cb = plt.colorbar(scat_plot) 

# plt.title('title')
# ax.set_xlabel('label for x axis') # notice the use of “ax” instead of “plt”
# ax.set_ylabel('label for y')
# ax.set_zlabel('z')
















# _____________________________________________________


# o


# def policy(Horizon,num_seq,env):
#     #num_seq is how many different randoms paths are to be taken
#     #Horizon is how far to look into future

#     #build model predictive controller (MPC)
#     aa=[] #vector of info from each random run
#     all_reward=[] #vector of rewards
#     all_rew_sum=[]
#       for i in range(Horizon):
#           action_seq=[random.uniform(-1,0),random.uniform(0,1)]
#           #random set of actions for the given path


#           current_state,new_state,rew,done,_=env.step(action_seq)
#           aa.append([current_state,new_state,rew,action_seq])
#           all_reward.append(rew)

#           #each aa of num_seq has next_state as ss of previous index

#           #want to keep action that corresponds to highest reward


#       rew_max=max(all_reward) #max reward
#       rew_min=min(all_reward)
#     #print([aa[0][0][0:2],aa[1][0][0:2]])
#     #print('\n')
#     #print(type(all_reward))
#     #print(aa[0][0])
#    # print(all_reward)
#     max_ind=all_reward.index(rew_max) #index of max reward
#     #print(max_ind)
#    # print('\n sum of rewards is ')
#     best_action=aa[max_ind][3] #gives action of aa with max reward
#    # print(best_action)
#     all_reward=[all_reward[0][0],all_reward[1][0],all_reward[2][0],all_reward[3][0],all_reward[4][0], \
#       all_reward[5][0],all_reward[6][0],all_reward[7][0],all_reward[8][0],all_reward[9][0]]
#     rew_sum=sum(all_reward) #sum of each array of awards
#    # print(rew_sum)

#    # print(aa[max_ind][1])


#     return rew_max, rew_min, rew_sum


# plt.figure()                        # Generate figure window

# env = SpacecraftDockingContinuous() # Call environment script
# s0 = env.reset()                    # instantiate the envirionment, i.e. collect the initial conditions

# # all_rew_sum=[]
# for i in range(30):               # Arbitrarily loop through a bunch of steps for rendering purposes
#     #sets time of simulation
    
#     rew_max,rew_min,rew_sum, action = policy(10,3,env,state)       #gives max reward (return of policy)

#     current_state,new_state, _, _, _ = env.step(a) # plug actions into agent, collect next states

#     if i % 1 == 0:                  # step sizes are small so we are showing every 8th step. Can change this number to make things run faster/slower
#         env.render(mode='human')


# env.close() #to fix python is likely shutting down" problem


# _____________________________________________________





# #original RunningThings.py
#OG RUNNING THINGS
# #before changing with MPC

# from envs.docking import SpacecraftDockingContinuous
# import matplotlib.pyplot as plt
# import numpy as np


# def policy():
#     '''choose action with uniform probability
#     Returns:
#         a (np.array| size action dim): selected action
#     '''
#     num_sequences = 1
#     ac_dim = 2
#     high = 1
#     low = -1

#     return np.random.rand(num_sequences, ac_dim) * (high - low) + low
#     #returns random 1x2 array

# plt.figure()                        # Generate figure window

# env = SpacecraftDockingContinuous() # Call environment script
# s0 = env.reset()                    # instantiate the envirionment, i.e. collect the initial conditions
# xc=0
# for i in range(2):               # Arbitrarily loop through a bunch of steps for rendering perposes
#     #sets how long simulation runs for
#     a = policy()                    # Collect arbitrary actions to make the spacecraft move
#     #random 1x2 array

#     state, _, _, _ = env.step(a) # plug actions into agent, collect next state
#     if i==0:
#         initstate=state
#     #print(state)
#     #print(a)
#     xc+=1
#     if i % 8 == 0:                  # step sizes are small so we are showing every 8th step. Can change this number to make things run faster/slower
#         env.render(mode='human')

# #print(initstate)
# #print(state)
# #print(xc)
# finalstate=state
# #print('final state is',finalstate)
# deltax=finalstate[0]-initstate[0]
# deltay=finalstate[1]-initstate[1]
# deltavec=[deltax,deltay]
# #print(deltavec)


# env.close()






