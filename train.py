# Adapted from Sourish Kundu's video : https://www.youtube.com/watch?v=_gmQZToTMac

import gym_super_mario_bros
import numpy as np
import torch
from nes_py.wrappers import JoypadSpace

from common import Common, Logger, Tracker
from src.wrapper import apply_wrappers
from src.DQNN.dqnn_agent import DQNNAgent

from pathlib import Path


class Train:
    logger = Logger()
    common = Common()
    tracker = Tracker(logger)

    def __init__(self, from_episode=0):
        env = gym_super_mario_bros.make(self.common.ENV_NAME)
        env.metadata['render.modes'] = ['human', 'rgb_array']
        env.metadata['apply_api_compatibility'] = True

        env = JoypadSpace(env, self.common.RIGHT_RUN)
        env = apply_wrappers(env)
        agent = DQNNAgent(env.observation_space.shape, env.action_space.n, self.common, self.logger)
        self.load_agent_state(agent, from_episode)

        state = env.reset()
        agent.debug_nn_size(state)

        for episode in range(from_episode, self.common.NUM_OF_EPISODES + 1):
            print(f'Episode {episode} started')
            done = False
            state = env.reset()
            self.tracker.init_reward()
            # print("### env reset, state: ", state)

            while not done:
                action = agent.choose_action(state)
                # print("action: ", action)
                next_state, reward, done, info = env.step(action)
                self.tracker.store_action(action, reward, episode)
                agent.store_in_memory(state, action, reward, next_state, done)
                agent.learn(episode)
                state = next_state
                # env.render()

            # end of current episode
            self.logger.add_scalar("learning_rate", agent.scheduler.get_last_lr()[0], episode)
            agent.scheduler.step()

            self.save_agent_state(agent, episode)
            self.tracker.end_of_episode(info, episode, self.save_actions)
            self.logger.flush()

        # save final episode
        self.save_agent_state(agent, episode)
        env.close()
        self.logger.close()

    def save_agent_state(self, agent, episode):
        if episode % self.common.SAVE_FREQ == 0:
            path = Path(self.logger.checkpoint_dir, f"agent_checkpoint_{episode}.pt")
            if path.exists():
                print(f"WARNING save_agent_state: path already exists, skipping save: {path}")
            else:
                agent.save_state(path)

    def save_actions(self, actions, episode, rewards):
        path = Path(self.logger.actions_dir, f"agent_actions_ep:{episode}_rw:{int(rewards)}.pt")
        if not self.logger.actions_dir.exists():
            print(f"WARNING save_actions: path doesn't exists, skipping save: {path}")
        else:
            torch.save(np.array(actions), path)

    def load_agent_state(self, agent, episode):
        path = Path(self.logger.checkpoint_dir, f"agent_checkpoint_{episode}.pt")
        if path.exists():
            agent.load_state(path)


if __name__ == "__main__":
    Train(0)
