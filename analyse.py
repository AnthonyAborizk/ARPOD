''' 
Things we want to do here: 
- load saved rl model and watch the algorithm play 
- load log data from training environment and analyze it
'''
import pickle 
import numpy as np 
import matplotlib.pyplot as plt
from torch import seed
from copy import deepcopy
import time 

#* LOAD CTF-LOGS 
data = []
filepath = 'data/log_11-09-2022_16-30'
with open(filepath, 'rb') as file: 
    data.append(pickle.load(file))

data = data[0]

# The observation space is defined as 
# Team  agent   alive   fuel    flagged     transfering     orbital     angle
# The data file is organized as follows 
# base0 obs; agent0 obs; base1 obs; agnet1 obs; actions 
# since the state space is 8 units long, for a 1v1 game there are 34 units per timestep

#* DATA EXTRACTION
dim = 0
# Agent0 dat a 
team        = data[:, dim+0]
agent       = data[:, dim+1]
alive       = data[:, dim+2]
fuel        = data[:, dim+3]
flagged     = data[:, dim+4]
transfering = data[:, dim+5]

#* PLOTS
tstep = np.array(range(data.shape[0])) 

plt.figure()
plt.plot(tstep, team)
plt.xlabel('Time Step (hrs)')
plt.ylabel('xpos')

plt.figure()
plt.plot(tstep, agent)
plt.xlabel('Time Step (hrs)')
plt.ylabel('ypos')

plt.figure()
plt.plot(team, agent)
plt.xlabel('xpos')
plt.ylabel('ypos')
plt.xlim(-10000 , 10000)
plt.ylim(-10000 ,10000)

plt.figure()
plt.plot(tstep, alive)
plt.xlabel('Time Step (hrs)')
plt.ylabel('xdot')

plt.figure()
plt.plot(tstep, fuel)
plt.xlabel('Time Step (hrs)')
plt.ylabel('ydot')

plt.figure()
plt.plot(tstep, flagged)
plt.xlabel('Time Step (hrs)')
plt.ylabel('action _ x')

plt.figure()
plt.plot(tstep, transfering)
plt.xlabel('Time Step (hrs)')
plt.ylabel('action _ y')

plt.figure()
plt.plot(tstep, np.sqrt(transfering**2 + flagged**2))
plt.show()

##################
#* PLAY AGENT
#################
from envs.docking import SpacecraftDockingContinuous

num_options = 2
max_history = 1000
seed = 42
loadmodel = 'models/optioncriticctf20220822-2252.txt'
max_steps_ep = 700

env = CTFENV(team_size=1, opponent='RandomPlayer', logging=False)
option_critic = optioncritic(in_features=32, num_actions=env._team_size*2+7+5, num_options=num_options)

# Create a prime network for more stable Q values
option_critic_prime = deepcopy(option_critic)

buffer = ReplayBuffer(capacity=max_history, seed=seed)  


option_critic.load_state_dict(torch.load(loadmodel))
option_critic_prime.load_state_dict(torch.load(loadmodel))

steps = 0;

# render_trained_model
reward = 0 ; option_lengths = {opt:[] for opt in range(num_options)}

obs   =  env.reset()
state =  option_critic.get_state(to_tensor(obs))
greedy_option  =  option_critic.greedy_option(state, env._team_size)
current_option = 0

done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0
while not done and ep_steps < max_steps_ep:
    epsilon =  option_critic.epsilon

    if option_termination:
        option_lengths[current_option].append(curr_op_len)
        current_option = np.random.choice(num_options) if np.random.rand() < epsilon else greedy_option
        curr_op_len = 0

    action, logp, entropy =  option_critic.get_action(state, current_option)

    next_obs, reward, done, _ =  env.step(action)
    buffer.push(obs, current_option, reward, next_obs, done)

    old_state = state
    state =  option_critic.get_state(to_tensor(next_obs))

    option_termination, greedy_option =  option_critic.predict_option_termination(state, current_option)

    # update global steps etc
    steps += 1
    ep_steps += 1
    curr_op_len += 1
    obs = next_obs
    env.render()
    time.sleep(.1)