import ray
# ray.init(num_cpus=1, num_gpus=0, log_to_driver=False, include_dashboard=False)
import gymnasium as gym
from gym import spaces
# from gym import spaces
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import MemoryTrackingCallbacks
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray import air
from envs.docking import SpacecraftDockingContinuous
import logging 
'''
def environment():

def training():

def rollouts():

def exploration():
'''
# num_rollout_workers=2, num_envs_per_worker=1 -- Can include these in .rollouts but idk how these work yet
'''
def env_creator(env_name):
    from envs.docking import SpacecraftDockingContinuous as env
    return env
'''
# env = env_creator()

# ray.tune.registry.register_env('ARPOD_PPO', lambda: config, env(config))

ray.init(num_cpus=1, num_gpus=0, log_to_driver=False, include_dashboard=False)
env = SpacecraftDockingContinuous(logdir = 'C:/Users/antho/OneDrive/Documents/ARPOD-Arthur/ARPOD/data')
register_env('spacecraft-docking-continuous-v0', lambda config: env)
config = PPOConfig().training(gamma = 0.9, lr=0.01)\
    .environment("spacecraft-docking-continuous-v0")\
    .rollouts(create_env_on_local_worker=True)\
    .callbacks(MemoryTrackingCallbacks)

pretty_print(config.to_dict())

# ray.rllib.utils.check_env([env])

algo = config.build()

algo.train()