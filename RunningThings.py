from envs.docking import SpacecraftDockingContinuous
import matplotlib.pyplot as plt
import numpy as np
#ldkfjl

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

for i in range(1000):               # Arbitrarily loop through a bunch of steps for rendering perposes
    a = policy()                    # Collect arbitrary actions to make the spacecraft move 
    state, _, _, _ = env.step(a[0]) # plug actions into agent, collect next state
    if i % 8 == 0:                  # step sizes are small so we are showing every 8th step. Can change this number to make things run faster/slower
        env.render(mode='human')
