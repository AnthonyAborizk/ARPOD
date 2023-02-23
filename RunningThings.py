import numpy as np
import matplotlib.pyplot as plt
from envs.docking import SpacecraftDockingContinuous
from scipy.signal import savgol_filter as sg
import math 

def policy(Horizon, num_seq, observation, env):
    low = env.action_space.low
    high = env.action_space.high
    action_mat=(np.random.uniform(low=low, high=high, size=(num_seq, Horizon, 3)))

    reward = np.zeros(num_seq)
    obs_old=np.tile(observation, (num_seq,1))       #collect current state from original observation
    act=action_mat[:, 0, :].reshape(num_seq, 3)     #Horizon x 3
    new_state = env.predict(obs_old, act)

    for j in range(1, Horizon):
        rew, _, _=env.get_reward(new_state,act, obs_old, j ) #collect returns from 
        act=action_mat[:,j,:].reshape(num_seq, 3 )            #Horizon x 2
        obs_old = new_state
        new_state = env.predict(obs_old, act)
        reward+=np.array(rew)               # sum reward for each different path
    best_index = reward.argmax()
    chosen_action=action_mat[best_index][0] # collect first action of path that has highest reward

    return chosen_action

# Generate figure window
logdir = 'data'

env = SpacecraftDockingContinuous(logdir)  

rew = 0
r = []
rho=[]
stuff = []
for i in range(1): 

    # instantiate the envirionment, i.e. collect the initial conditions
    state = env.reset()                       
    rH = env.rH
    # sets time of simulation in real life
    for k in range(15000):                     
        steps=k
        rho.append([state])
        # gives action with max reward (return of policy)
        chosen_action = policy(10,10000,state,env)
        stuff.append(chosen_action)
        # plug actions into agent, collect next states        
        state, reward, done, _ = env.step(chosen_action) 
        rew += reward['rew']
        r+= reward['rew'].tolist()

        if  k % 10  == 0:   
            env.render(mode='human')
        elif k>=15000 and k % 100 == 0: 
            env.render(mode='human')

        if done: 
            print('Done')
            print('reward: ', reward)
            print('state: ', state)
            break

env.close()

states = np.array(rho)

x = states[:,:,0]
y = states[:,:,1]
psi = states[:, : ,2]
xvel = states[:,:,3]
yvel = states[:,:,4]
psi_dot = states[:,:,5]

plt.figure()
plt.scatter( range(len(psi)), psi)
plt.plot([0, len(psi)], [0.25, 0.25], 'r--')
plt.plot([0, len(psi)], [-0.25, -0.25], 'r--')
# plt.plot(range(len(psi)), sg(psi.squeeze(), 11, 2 ))
plt.title('psi')

plt.figure()
plt.plot(r)
plt.title('Reward')

plt.figure()
plt.plot(np.array(stuff)[:, 2])
plt.plot([0, len(stuff)], [2*math.pi/180, 2*math.pi/180], 'r--')
plt.plot([0, len(stuff)], [-2*math.pi/180, -2*math.pi/180], 'r--')
plt.title('Actions')

plt.figure()
circle1 = plt.Circle((0, 0), 100, edgecolor='g', alpha=0.1)
plt.gca().add_patch(circle1)    
plt.plot([0, 100*math.tan(np.pi/6), -100*math.tan(np.pi/6), 0], [0, -100, -100, 0], color='orange', label='Line of Sight')
plt.plot(y, x)
plt.xlim([100, -100])
plt.xlabel('Local Horizontal [m]')
plt.ylabel('Local Vertical [m]')
plt.title('Trajectory')
plt.legend() 

plt.figure()
plt.plot(xvel)
plt.title('xvelocity')
plt.figure()
plt.plot(yvel)
plt.title('yvelocity')

plt.figure()
plt.plot(psi)
plt.title('psi')

plt.figure()
plt.plot(psi_dot)
plt.title('psi_dot')
plt.show()
