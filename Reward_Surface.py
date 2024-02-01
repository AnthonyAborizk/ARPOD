# The goal of this script is to generate a surface plot based on extrapolated reward data. 
# in other words a surface plot based on the reward function written. 


#?  Phase 3 reward
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import math
import matplotlib.pyplot as plt
import random
from envs.docking import ARPOD


def get_reward(xpos, ypos, xd, yd):

    hstep = 0
    env = ARPOD()
    rho = np.linalg.norm([xpos,ypos], axis=0)
    
    vH = np.linalg.norm([xd, yd], axis=0)  # Velocity Magnitude
    vH_max = 2 * env.N * rho + env.VEL_THRESH           # Max Velocity
    vH_min = 1/2 * env.N * rho - env.VEL_THRESH         # Min Velocity
    # if env.phase == 3: 

    dones = np.zeros((xpos.shape[0],))   # initialize dones vector
    dn1 = np.zeros((xpos.shape[0],))
    dn2 = np.zeros((xpos.shape[0],))

    dn1[abs(ypos) <= env.pos_threshold] = 1
    dn2[abs(xpos) <= env.pos_threshold] = 1
    dones[dn1+dn2>1] = 1 # if s/c is within docking bounds
    dones[abs(xpos) > env.x_threshold] = 1  # out of bounds #! removed for hrl
    dones[abs(ypos) > env.y_threshold] = 1  # out of bounds #! REMOVED FOR HRL
    # dones[(env.steps+hstep) * env.TAU >= env.max_time-1] = 1

    c = np.zeros([xpos.shape[0], 3])
    c[:, 1] = 100
    position = np.array([xpos, ypos, np.zeros(xpos.shape[0])]).transpose()
    val = position[:, 1] * 100 / (100 * rho)  # dot(position, c) / mag(position)*mag(c)

    reward = (-rho)/50 * env.TAU # reward for getting closer to target
    if ~all(dones):    
        reward[((dones==0) & (vH < vH_min))] += -0.0075*abs(vH[((dones==0) & (vH < vH_min))]-vH_min[((dones==0) & (vH < vH_min))]) * env.TAU 
        reward[((dones==0) & (vH > vH_max))] += -0.0035*abs(vH[((dones==0) & (vH > vH_max))]-vH_max[((dones==0) & (vH > vH_max))]) * env.TAU
        reward[((dones==0) & (vH < 2*env.VEL_THRESH) & (vH < vH_min))] += -0.0075/2 * env.TAU
        reward[((dones==0) & (vH < 2*env.VEL_THRESH) & (vH > vH_max))] += -0.0075/2 * env.TAU
        reward[((dones==0) & (rho <= 100) & (val <= math.cos(env.theta_los)))] += -1

    if all(dones) != False: 
        reward[((dones==1) & (abs(xpos) <= env.pos_threshold) & (abs(ypos) <= env.pos_threshold) & (vH > env.VEL_THRESH))] += -0.001
        reward[((dones==1) & (abs(xpos) <= env.pos_threshold) & (abs(ypos) <= env.pos_threshold) & (vH <= env.VEL_THRESH))] += 10
        reward[((dones==1) & ((env.steps+hstep) * env.TAU > env.max_time))] += -1
        reward[((dones==1) & (abs(xpos) > env.x_threshold) | (abs(ypos) > env.y_threshold))] += -1  #! removed for hrl

    return reward, dones

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-25, 25, 1, dtype='int8')
xd = yd = np.arange(-20, 20, .8, dtype='int8')

X, Y, Xd, Yd = np.meshgrid(x, y, xd, yd)

zs, dones = np.array(get_reward(np.ravel(X), np.ravel(Y), np.ravel(Xd), np.ravel(Yd)))
Z = zs.reshape(X.shape)

fig,ax=plt.subplots(1,1)
cp = ax.contourf(X[:, :, 0, 0], Y[:, :, 0, 0], Z[:, :, 0, 0])
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
#ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')

fig2,ax2=plt.subplots(1,1)
i = 2
cp2 = ax2.contourf(Xd[i, i,:, :], Yd[i, i,:, :], Z[i, i,:, :])
fig2.colorbar(cp2) # Add a colorbar to a plot

plt.show()


def get_reward2(xpos, ypos, xd, yd):

    hstep = 0
    env = ARPOD()
    rho = np.linalg.norm([xpos,ypos], axis=0)
    vH = np.linalg.norm([xd, yd], axis=0)  # Velocity Magnitude
    vH_max = 2 * env.N * rho + env.VEL_THRESH           # Max Velocity
    vH_min = 1/2 * env.N * rho - env.VEL_THRESH         # Min Velocity
    c        = np.zeros([xpos.shape[0], 3])
    c[:, 1]  = 100
    position = np.array([xpos, ypos, np.zeros(xpos.shape[0])]).transpose()
    val      = position[:, 1] * 100 / (100 * rho)  # dot(position, c) / mag(position)*mag(c)
    
    dones = np.zeros((xpos.shape[0],))

    dones[(rho <= 100) & (val > math.cos(env.theta_los))] = 1
    dones[abs(xpos) > env.x_threshold] = 1
    dones[abs(ypos) > env.y_threshold] = 1
    # dones[(env.steps+hstep) * env.TAU >= env.max_time-1] = 1 # if the time limit has been exceeded 

    reward = (-1 - rho + rH_old)/2000 * env.TAU

    # if rH_old.shape[0] < 1000: 
        # reward[(rho>100)] = (-1 - rho[(rho>100)] + rH_old)/2000 * env.TAU
    # else: 
        # reward[(rho>100)] = (-1 - rho[(rho>100)] + rH_old[(rho>100)])/2000 * env.TAU

    if ~all(dones):  
        reward[((dones==0) & (vH < vH_min))] += -0.0075*abs(vH[((dones==0) & (vH < vH_min))]-vH_min[((dones==0) & (vH < vH_min))]) * env.TAU
        reward[((dones==0) & (vH > vH_max))] += -0.0035*abs(vH[((dones==0) & (vH > vH_max))]-vH_max[((dones==0) & (vH > vH_max))]) * env.TAU
        reward[((dones==0) & (rho <= 100) & (val <= math.cos(env.theta_los)))] += -.9 #! uncomment this
        # reward[((dones==0) & (rho <= 200) & (val <= math.cos(env.theta_los)) & (ypos > ypos_old[:, 1]) ) ] += .0001 #! uncomment this

    if all(dones) != False: 
        reward[((dones==1) & (rho <= 100) & (val > math.cos(env.theta_los)))]  += 1 # success #! old 
        reward[((dones==1) & ((env.steps+hstep) * env.TAU > env.max_time))] += -1
        reward[((dones==1) & (abs(xpos) > env.pos_threshold) & (abs(ypos) > env.pos_threshold))] += -1

    return reward