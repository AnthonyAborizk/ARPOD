import numpy as np
import pytest

import gym
from gym import spaces
from gym.wrappers import FlattenObservation


@pytest.mark.parametrize("env_id", ["Blackjack-v1"])
def test_flatten_observation(env_id):
    env = gym.make(env_id)
    wrapped_env = FlattenObservation(env)

    obs = env.reset()
    wrapped_obs = wrapped_env.reset()

    space = spaces.Tuple((spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2)))
    wrapped_space = spaces.Box(0, 1, [32 + 11 + 2], dtype=np.int64)

    assert space.contains(obs)
    assert wrapped_space.contains(wrapped_obs)
