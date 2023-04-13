import ray
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
from ray.rllib.utils.gym import check_old_gym_env, try_import_gymnasium_and_gym, convert_old_gym_space_to_gymnasium_space


env = SpacecraftDockingContinuous(logdir = 'C:/Users/antho/OneDrive/Documents/ARPOD-Arthur/ARPOD/data')
register_env('spacecraft-docking-continuous-v0', lambda config: env)
## env = convert_old_gym_space_to_gymnasium_space(env)
ray.rllib.utils.check_env(env) # NO BRACKET