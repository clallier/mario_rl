# Adapted from Sourish Kundu's video : https://www.youtube.com/watch?v=_gmQZToTMac

import gym_super_mario_bros
import numpy as np
import torch
from nes_py.wrappers import JoypadSpace

from common import Common
from wrapper import apply_wrappers
from agent import Agent
import collections
from statistics import mean, stdev
from pathlib import Path

class Train:
    common = Common()

    def __init__(self, from_episode=0):
        env = gym_super_mario_bros.make(self.common.ENV_NAME)
        env.metadata['render.modes'] = ['human', 'rgb_array']
        env.metadata['apply_api_compatibility'] = True
        
        env = JoypadSpace(env, self.common.RIGHT_RUN)
        env = apply_wrappers(env)
        agent = Agent(env.observation_space.shape, env.action_space.n, self.common)
        self.load_agent_state(agent, from_episode)

        state = env.reset()
        agent.debug_nn_size(state)

        episode = 0
        best_reward = 0
        d = collections.deque(maxlen=10)
        flag_get_sum = 0

        for episode in range(from_episode, self.common.NUM_OF_EPISODES+1):
            print(f'Episode {episode} started')
            done = False
            state = env.reset()
            rewards = 0
            actions = []
            info = {}
            # print("### env reset, state: ", state)

            while not done:
                action = agent.choose_action(state)
                actions.append(action)
                # print("action: ", action)
                next_state, reward, done, info = env.step(action)
                rewards += reward['raw']
                agent.store_in_memory(state, action, reward['normalized'], next_state, done)
                agent.learn(episode)
                state = next_state

                self.common.add_scalar("reward_raw", reward['raw'], episode)
                self.common.add_scalar("reward_normalized", reward['normalized'], episode)
                # env.render()

            # end of current episode
            self.common.add_scalar("learning_rate", agent.scheduler.get_last_lr()[0], episode)
            agent.scheduler.step()

            d.append(rewards)
            flag_get_sum += info.get('flag_get', False) if info is not None else False
            avg_10 = mean(d) if len(d) > 2 else -1
            std_10 = stdev(d) if len(d) > 2 else -1
            self.common.add_scalar("rewards", rewards, episode)
            self.common.add_scalar("avg_10", avg_10, episode)
            self.common.add_scalar("std_10", std_10, episode)
            self.common.add_scalar("flag_get_sum", flag_get_sum, episode)
            print(f'Episode {episode}, rewards: {rewards}, best so far: {best_reward}, '
                f'mean_10: {avg_10:.1f}, std_10: {std_10:.1f}')
            self.save_agent_state(agent, episode)
            if rewards > best_reward:
                best_reward = rewards
                self.save_actions(actions, episode, rewards)
            self.common.writer.flush()

        # save final episode
        self.save_agent_state(agent, episode)
        env.close()
        self.common.writer.close()

    def save_agent_state(self, agent, episode):
        if episode % self.common.SAVE_FREQ == 0:
            path = Path(self.common.checkpoint_dir, f"agent_checkpoint_{episode}.pt")
            if path.exists():
                print(f"WARNING: path already exist, skipping save: {path}")
            else:
                agent.save_state(path)

    def save_actions(self, actions, episode, rewards):
        path = Path(self.common.actions_dir, f"agent_actions_ep:{episode}_rw:{int(rewards)}.pt")
        torch.save(np.array(actions), path)


    def load_agent_state(self, agent, episode):
        path = Path(self.common.checkpoint_dir, f"agent_checkpoint_{episode}.pt")
        if path.exists():
            agent.load_state(path)


if __name__ == "__main__":
    Train(0)
