""" import ray
ray.init()
from ray import tune

tune.run("PPO", 
        config = "CartPole-v1",
        ) """

""" from ray.rllib.algorithms.ppo import PPOConfig
config = PPOConfig()  
config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)  
config = config.resources(num_gpus=0)  
config = config.rollouts(num_rollout_workers=4)  
print(config.to_dict())  
# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build(env="CartPole-v1")  
algo.train()   """

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from envs.docking import SpacecraftDockingContinuous
from ray.tune.registry import register_env

env = SpacecraftDockingContinuous(logdir = 'C:/Users/antho/OneDrive/Documents/ARPOD-Arthur/ARPOD/data')
register_env('spacecraft-docking-continuous-v0', lambda config: env)

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env = 'spacecraft-docking-continuous-v0')
    .build()
)

for i in range(1000000):  # 1 million itertions -- leave overnight
    result = algo.train()
    print(pretty_print(result))

    if (i+1) % 2 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
