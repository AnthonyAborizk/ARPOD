from envs.docking.docking import SpacecraftDockingContinuous
from envs.docking.arpod import ARPOD 
from gym.envs.registration import register

register(
    id='spacecraft-docking-continuous-v0',
    entry_point='envs.docking:SpacecraftDockingContinuous',
    max_episode_steps=4000
)

register(
    id='arpod-v0',
    entry_point='envs.docking:ARPOD',
    max_episode_steps=4000
)