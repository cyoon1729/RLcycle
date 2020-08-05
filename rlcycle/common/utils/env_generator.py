import gym
from omegaconf import DictConfig

import pybulletgym  # noqa: F401
from rlcycle.common.utils.env_wrappers import (
    ClipRewardEnv,
    ImageToPyTorch,
    TimeLimit,
    make_atari,
    wrap_deepmind,
)


def generate_atari_env(env_info: DictConfig) -> gym.Env:
    """Generate atari env from given config"""
    assert env_info.is_atari is True
    env = make_atari(env_id=env_info.name, max_episode_steps=env_info.max_episode_steps)
    env = wrap_deepmind(env, frame_stack=env_info.frame_stack)
    env = ImageToPyTorch(env)
    return env


def generate_env(env_info: DictConfig) -> gym.Env:
    """Generate non-atari env from given config"""
    assert env_info.is_atari is False, "For atari envs use generate_atari_env()"
    env = gym.make(env_info.name)
    if env_info.max_episode_steps is not None:
        env = TimeLimit(env, env_info.max_episode_steps)
    if env_info.clip_rewards is not None:
        env = ClipRewardEnv(env)
    return env
