import numpy as np

import gym
from gym import spaces


class RescaleAction(gym.ActionWrapper):
    r"""Rescales the continuous action space of the environment to a range [min_action, max_action].

    The wrapped environment `env` must have an action space of type `spaces.Box`. If `min_action`
    or `max_action` are numpy arrays, the shape must match the shape of the environment's action space.

    Example::

        >>> RescaleAction(env, min_action, max_action).action_space == Box(min_action, max_action)
        True

    Args:
        env: The environment that will be wrapped
        min_action (Union[np.array, float]): The lower bound of the new action space. This may be a numpy array or a scalar.
        max_action (Union[np.array, float]): The upper bound of the new action space. This may be a numpy array or a scalar.
    """

    def __init__(self, env, min_action, max_action):
        assert isinstance(
            env.action_space, spaces.Box
        ), f"expected Box action space, got {type(env.action_space)}"
        assert np.less_equal(min_action, max_action).all(), (min_action, max_action)

        super().__init__(env)
        self.min_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + min_action
        )
        self.max_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + max_action
        )
        self.action_space = spaces.Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        assert np.all(np.greater_equal(action, self.min_action)), (
            action,
            self.min_action,
        )
        assert np.all(np.less_equal(action, self.max_action)), (action, self.max_action)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = np.clip(action, low, high)
        return action
