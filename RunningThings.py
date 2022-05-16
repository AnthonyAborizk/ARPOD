from envs.docking import SpacecraftDockingContinuous
import matplotlib.pyplot as plt
import numpy as np


def policy():
    '''choose action with uniform probability
    Returns:
        a (np.array| size action dim): selected action
    '''
    num_sequences = 1 
    ac_dim = 2
    high = 1
    low = -1

    return np.random.rand(num_sequences, ac_dim) * (high - low) + low

plt.figure()                        # Generate figure window

env = SpacecraftDockingContinuous() # Call environment script
s0 = env.reset()                    # instantiate the envirionment, i.e. collect the initial conditions
xc=0
for i in range(1000):               # Arbitrarily loop through a bunch of steps for rendering perposes
    #sets time of simulation  
    a = policy()                    # Collect arbitrary actions to make the spacecraft move 
    state, _, _, _ = env.step(a[0]) # plug actions into agent, collect next state
    if i==0:
        initstate=state
    #print(state)
    #print(a)
    xc+=xc+1
    if i % 8 == 0:                  # step sizes are small so we are showing every 8th step. Can change this number to make things run faster/slower
        env.render(mode='human')

#print(initstate)
#print(state)
#print(xc)
finalstate=state    
#print('final state is',finalstate)  
deltax=finalstate[0]-initstate[0]
deltay=finalstate[1]-initstate[1]
deltavec=[deltax,deltay]
#print(deltavec)


env.close()
       
       
       
        
# pip install matplotlib
# pip install numpy
# pip install pyglet
# conda activate env
# sudo git clone https://github.com/openai/gym
# pip install -e .
# export DISPLAY=172.26.128.1:0.0
# export LIBGL_ALWAYS_INDIRECT=1

# apt-get install python-opengl
# sudo apt-get install python-opengl

# conda create -n env python=3.9

#if "NoRushDisplayException: Cannot connect to "None"
#open xlaunch and all three including disable access control
#open Ubuntu and type cat /etc/resolv.conf
#export DISPLAY=172.26.128.1:0.0
#export LIBGL_ALWAYS_INDIRECT=1