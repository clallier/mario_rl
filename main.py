# Adapted from Sourish Kundu's video : https://www.youtube.com/watch?v=_gmQZToTMac

import gym_super_mario_bros
import numpy as np
import torch
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from wrapper import apply_wrappers
from agent import Agent
import collections
from statistics import mean, stdev
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

NUM_OF_EPISODES = 80_000
ENV_NAME = 'SuperMarioBros-1-1-v3'
CHECKPOINT_DIR = './checkpoints/'
ACTIONS_DIR = './actions/'
SAVE_FREQ = 1000  # in episode


def train(from_episode=0):
    writer = SummaryWriter()
    env = gym_super_mario_bros.make(ENV_NAME)
    env.metadata['render.modes'] = ['human', 'rgb_array']
    env.metadata['apply_api_compatibility'] = True

    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)
    agent = Agent(env.observation_space.shape, env.action_space.n)
    load_agent_state(agent, from_episode)

    state = env.reset()
    agent.debug_nn_size(state)

    episode = 0
    best_reward = 0
    d = collections.deque(maxlen=10)

    for episode in range(from_episode, NUM_OF_EPISODES):
        print(f'Episode {episode} started')
        done = False
        state = env.reset()
        rewards = 0
        actions = []
        # print("### env reset, state: ", state)

        while not done:
            action = agent.choose_action(state)
            actions.append(action)
            # print("action: ", action)
            next_state, reward, done, info = env.step(action)
            rewards += reward
            agent.store_in_memory(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            # env.render()

        # end of current episode
        d.append(rewards)
        flag_get = info.get('flag_get', False) if info is not None else False
        avg_10 = mean(d) if len(d) > 2 else -1
        std_10 = stdev(d) if len(d) > 2 else -1
        writer.add_scalar("rewards", rewards, episode)
        writer.add_scalar("avg_10", avg_10, episode)
        writer.add_scalar("std_10", std_10, episode)
        writer.add_scalar("flag_get", flag_get, episode)
        print(f'Episode {episode}, rewards: {rewards}, best so far: {best_reward}, '
              f'mean_10: {avg_10:.1f}, std_10: {std_10:.1f}')
        save_agent_state(agent, episode)
        if rewards > best_reward:
            best_reward = rewards
            save_actions(actions, episode, rewards)
        writer.flush()

    # save final episode
    save_agent_state(agent, episode)
    env.close()
    writer.close()


def test_from_checkpoint(episode_n):
    env = gym_super_mario_bros.make(ENV_NAME)
    env.metadata['render.modes'] = ['rgb_array']
    env.metadata['apply_api_compatibility'] = True

    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)
    agent = Agent(env.observation_space.shape, env.action_space.n)
    agent.load_state(episode_n)

    done = False
    state = env.reset()
    reward_sum = 0
    print("### env reset, state: ", state)

    while not done:
        action = agent.choose_action(state)
        print("action: ", action)
        next_state, reward, done, info = env.step(action)
        reward_sum += reward
        state = next_state
        rgb_array = env.render(
            mode="rgb_array",
        )
        print(rgb_array)

    print("### reward_sum: ", reward_sum)
    env.close()


def test_from_actions(episode, rewards):
    actions = load_actions(episode, rewards)

    env = gym_super_mario_bros.make(ENV_NAME)
    env.metadata['render.modes'] = ['rgb_array']
    env.metadata['apply_api_compatibility'] = True

    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)

    state = env.reset()
    reward_sum = 0
    print("### env reset, state: ", state)

    for action in actions:
        print("action: ", action)
        next_state, reward, done, info = env.step(action)
        reward_sum += reward
        rgb_array = env.render()
    print("### reward_sum: ", reward_sum)
    env.close()


def save_agent_state(agent, episode):
    if episode % SAVE_FREQ == 0:
        path = Path(CHECKPOINT_DIR, f"agent_checkpoint_{episode}.pt")
        if path.exists():
            print(f"WARNING: path already exist, skipping save: {path}")
        else:
            agent.save_state(path)


def save_actions(actions, episode, rewards):
    path = Path(ACTIONS_DIR, f"agent_actions_ep:{episode}_rw:{int(rewards)}.pt")
    torch.save(np.array(actions), path)


def load_agent_state(agent, episode):
    path = Path(CHECKPOINT_DIR, f"agent_checkpoint_{episode}.pt")
    if path.exists():
        agent.load_state(path)


def load_actions(episode, rewards):
    path = Path(ACTIONS_DIR, f"agent_actions_ep{episode}_rw{rewards}.pt")
    actions = torch.load(path)
    return actions


if __name__ == "__main__":
    train(0)
    # test_from_checkpoint(58000)
    # test_from_actions(58000, 242)
