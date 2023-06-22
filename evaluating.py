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


# agent = ppo.PPOTrainer(algo, env= 'spacecraft-docking-continuous-v0')
algo.restore('c:/Users/antho/ray_results/PPO_spacecraft-docking-continuous-v0_2023-06-08_01-52-06sr91n78x/checkpoint_001100')

episode_reward = 0
done = False
obs,_ = env.reset()
step = 0
for i in range(10000):
    action = algo.compute_single_action(obs)
    obs, reward, done, info,_ = env.step(action)
    episode_reward += reward
    if step % 1 == 0:
        env.render()
    step += 1