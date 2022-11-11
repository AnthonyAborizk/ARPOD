from hashlib import new
from envs.docking import SpacecraftDockingContinuous
import matplotlib.pyplot as plt
import numpy as np
import time
startTime = time.time()


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

plt.figure()                               # Generate figure window
logdir = 'data'
env = SpacecraftDockingContinuous(logdir)  # Call environment script
state = env.reset()                        # instantiate the envirionment, i.e. collect the initial conditions
rew = 0
for k in range(15000):                     # sets time of simulation in real life
    steps=k
       
    chosen_action = policy(10,1000,state,env)        # gives action with max reward (return of policy)
    state, reward, done, _ = env.step(chosen_action) # plug actions into agent, collect next states
    rew += reward['rew']
    if k < 800 and k % 100 == 0:    # step sizes are small so we are showing every 8th step
                                    # Can change this number to make things run faster/slower
                                    # controls size of step in simulation
        env.render(mode='human')
    elif k>=800 and k % 150 == 0: 
        env.render(mode='human')
    if done: 
        print('Done')
        print(reward)
        break

env.close()                            #to fix "python is likely shutting down" problem
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

print('reward: ' , rew)






