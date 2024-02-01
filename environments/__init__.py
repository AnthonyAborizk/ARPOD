from environments.actuated3dof import ActuatedDocking3DOF
from environments.actuated6dof import ActuatedDocking6DOF
from gymnasium.envs.registration import register

register(
    id='underactuated-v0',
    entry_point='environments:UnderactuatedDocking',
    max_episode_steps=4000
)

register(
    id='actuated-v0',
    entry_point='environments:ActuatedDocking',
    max_episode_steps=4000
)