
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.utils import check_env
import time 
from aero_gym.envs.spacecraft_docking import SpacecraftDockingContinuous

print("imported!")

select_env = "spacecraft-docking-continuous-v0"
register_env(select_env, lambda config: SpacecraftDockingContinuous())

alg = "ppo"
num_cpus = 1
num_gpus = 0
num_workers = 1
num_envs_per_worker = 1
rollout_fragment_length = 43500
train_batch_size = rollout_fragment_length * num_workers
batch_mode = "complete_episodes" #could be "truncate_episodes" or "complete_episodes"

ray.init(num_cpus=num_cpus, num_gpus=num_gpus, log_to_driver=False, include_dashboard=False)
config = ppo.DEFAULT_CONFIG.copy()
config.update({"num_workers":num_workers, "num_envs_per_worker":num_envs_per_worker, "rollout_fragment_length":rollout_fragment_length, "train_batch_size":train_batch_size,"batch_mode":batch_mode})
config["num_cpus_for_driver"] = 1
config["num_cpus_per_worker"] = 1 

agent = ppo.PPOTrainer(config, env=select_env)
agent.restore('anth_model_t2328_08182022_ppo_workers1_cpu1_gpu0/checkpoint_000172/checkpoint-172')

env = SpacecraftDockingContinuous()
episode_reward = 0
done = False
obs = env.reset()
step = 0
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    if step % 8 == 0:
        env.render()
    step += 1