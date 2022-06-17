from envs.docking.docking import SpacecraftDockingContinuous
from gym.envs.registration import register

register(
    id='spacecraft-docking-continuous-v0',
    entry_point='envs.docking:SpacecraftDockingContinuous',
    max_episode_steps=4000
)
