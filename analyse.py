''' 
Things we want to do here: 
- load saved rl model and watch the algorithm play 
- load log data from training environment and analyze it
'''
import pickle 
import numpy as np 
import matplotlib.pyplot as plt
# from torch import seed
# from copy import deepcopy
import time 
from scipy.signal import savgol_filter as sg
import math

#* LOAD CTF-LOGS 
data = []
filepath = '/home/aborizk_anthony/Orbital_Debris/ARPOD/data/log_02-02-2023_14-47'
# filepath = 'data/hrlFirstRun'
with open(filepath, 'rb') as file: 
    data.append(pickle.load(file))

data = data[0]

# The observation space is defined as 
# Team agent alive fuel flagged transfering orbital angle
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
# tstep = np.array(range(1547))
plt.figure()
plt.plot(tstep, team)
plt.xlabel('Time Step (hrs)')
plt.ylabel('xpos')

plt.figure()
plt.plot(tstep, agent)
plt.xlabel('Time Step (hrs)')
plt.ylabel('ypos')

plt.figure()
circle1 = plt.Circle((0, 0), 100, edgecolor='g', alpha=0.1)
plt.gca().add_patch(circle1)    
plt.plot([0, -100*math.tan(np.pi/6), 100*math.tan(np.pi/6), 0], [0, 100, 100, 0], color='orange', label='Line of Sight')
plt.scatter(team, agent, s=1, label='Docking Trajectory')
plt.xlabel('Local Horizontal [m]')
plt.ylabel('Local Vertical [m]')
# plt.xlim(-105, 105)
# plt.ylim(-105, 105)
plt.legend()

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
# plt.plot(tstep, np.sqrt(transfering**2 + flagged**2), legend='Raw Action Input')
# plt.plot(tstep, sg(np.sqrt(transfering**2 + flagged**2), 101, 2 ), legend='')
plt.xlabel('Time Step [s] ')
plt.ylabel('Thrust Magnitude [N]')
plt.show()