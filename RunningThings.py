import time
import math
import numpy as np
from hashlib import new
import matplotlib.pyplot as plt
# from envs.docking import ARPODContinuous
from envs.docking import SpacecraftDockingContinuous
startTime = time.time()

def hrl_policy(Horizon, num_seq, observation, env):

    low = [0, 0]
    high = [1, 5]
    action_mat=(np.random.uniform(low=low, high=high, size=(num_seq, Horizon, 2)))
    reward = np.zeros(num_seq)
    obs_old=np.tile(observation, (num_seq,1))       #collect current state from original observation
    act=action_mat[:, 0,:].reshape(num_seq, 2)      #Horizon x 2

    new_state = env.predict(obs_old, act)

    for j in range(1, Horizon):
        rew, _, _=env.get_reward(new_state,act, obs_old, j ) #collect returns from 
        act=action_mat[:,j,:].reshape(num_seq, 2)            #Horizon x 2
        obs_old = new_state
        new_state = env.predict(obs_old, act)
        reward+=np.array(rew)               # sum reward for each different path
    best_index = reward.argmax()
    chosen_action=action_mat[best_index][0] # collect first action of path that has highest reward

    return chosen_action

def policy(Horizon, num_seq, observation, env):
    # num_seq is how many different randoms paths are to be taken
    # Horizon is how far to look into future in each path
    low = env.action_space.low
    high = env.action_space.high
    action_mat=(np.random.uniform(low=low, high=high, size=(num_seq, Horizon, 2)))
    #random Horizon x num_seq x 2 3D matrix of actions
    # build model predictive controller (MPC)
    reward = np.zeros(num_seq)
    obs_old=np.tile(observation, (num_seq,1))       #collect current state from original observation
    # obs_old = env.state
    act=action_mat[:, 0,:].reshape(num_seq, 2)     #Horizon x 2
    new_state = env.predict(obs_old, act)

    for j in range(1, Horizon):
        rew, _, _=env.get_reward(new_state,act, obs_old, j ) #collect returns from 
        act=action_mat[:,j,:].reshape(num_seq, 2)            #Horizon x 2
        obs_old = new_state
        new_state = env.predict(obs_old, act)
        reward+=np.array(rew)               # sum reward for each different path
    best_index = reward.argmax()
    chosen_action=action_mat[best_index][0] # collect first action of path that has highest reward

    return chosen_action

# Generate figure window
plt.figure()                               
logdir = 'data'

# Call environment script #! old
env = SpacecraftDockingContinuous(logdir)  
# env = ARPODContinuous(logdir)                        #! new for hrl

rew = 0
r = []
r1, r2, r3 = [],[],[]
rho, rho0=[],[]
for i in range(10): 

    # instantiate the envirionment, i.e. collect the initial conditions
    state = env.reset()                       
    rH = env.rH
    # sets time of simulation in real life
    for k in range(15000):                     
        steps=k

        if rH > 1000: #m
            angle = np.arctan2(state[1], state[0])
            
        # gives action with max reward (return of policy)
        chosen_action = policy(5,10000,state,env)

        # plug actions into agent, collect next states        
        state, reward, done, _ = env.step(chosen_action) 
        rew += reward['rew']
        r+= reward['rew'].tolist()

        # if len(reward['r1'])==0: 
        #     r1+=[0]
        # else: 
        #     r1+= reward['r1']
        # if len(reward['r2'])==0:
        #     r2+=[0]
        # else: 
        #     r2+= reward['r2']
        # if len(reward['r3'])==0:
        #     r3+=[0]
        # else: 
        #     r3+= reward['r3']
        # rho += [reward['rho'] - reward['rho0']]
        if  k % 300 == 0:    # step sizes are small so we are showing every 8th step
                                        # Can change this number to make things run faster/slower
                                        # controls size of step in simulation
            env.render(mode='human')
        elif k>=15000 and k % 100 == 0: 
            env.render(mode='human')

        if done: 
            print('Done')
            print(reward)
            break

env.close()
plt.plot(r, label='total')
plt.plot(r1, label='dense')
plt.plot(r2, label='vmin')
plt.plot(r3, label='vmax')
plt.plot(rho, label='diff')
plt.legend()
plt.show()

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

print('reward: ' , rew)






