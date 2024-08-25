# Adapted from Sourish Kundu's video : https://www.youtube.com/watch?v=_gmQZToTMac

import numpy as np
import torch
from src.Common.common import Common, Logger, Tracker
from src.Common.conv_calc import debug_count_params, debug_nn_size
from src.Common.async_single_sim import AsyncSingleSim
from src.DQNN.dqnn_agent import DQNNAgent

from pathlib import Path


class DQNNTrainer:
    def __init__(self, common, from_episode=0):
        self.common = common
        self.logger = Logger(common)
        self.tracker = Tracker(self.logger)

        # DQNN config
        dqnn_config_path = Common.get_file(f"./config_files/{common.config.get('dqnn_config_file')}")
        self.dqnn_config = common.load_config_file(dqnn_config_path)
        self.logger.add_file(dqnn_config_path)

        self.num_episodes = self.dqnn_config.get("NUM_OF_EPISODES")
        self.save_freq = self.dqnn_config.get("SAVE_FREQ")

        sim = AsyncSingleSim(common)
        agent = self.create_agent(sim)
        self.load_agent_state(agent, from_episode)

        state = sim.reset()
        debug_nn_size(agent.online_network.conv, state)
        debug_count_params(agent.online_network.network)

        for episode in range(from_episode, self.num_episodes + 1):
            print(f'Episode {episode} started')
            done = False
            state = sim.reset()
            self.tracker.init_reward()
            # print("### env reset, state: ", state)

            while not done:
                action = agent.choose_action(state)
                # print("action: ", action)
                next_state, reward, done, info = sim.step(action)
                self.tracker.store_action(action, info, agent.learn_step_counter)
                agent.store_in_memory(state, action, info, next_state, done)
                agent.learn(episode)
                state = next_state
                # env.render()

            # end of current episode
            current_lr = agent.scheduler.get_last_lr()[0]
            self.logger.add_scalar("learning_rate", current_lr, agent.learn_step_counter)
            agent.scheduler.step()

            self.save_agent_state(agent, episode)
            self.tracker.end_of_episode(info, episode, self.save_actions)
            self.logger.flush()

        # save final episode
        self.save_agent_state(agent, episode)
        sim.close()
        self.logger.close()

    def create_agent(self, sim):
        input_dims = sim.single_observation_space.shape
        output_dims = sim.single_action_space.n
        return DQNNAgent(input_dims, output_dims, self.dqnn_config, self.common, self.logger)

    def save_agent_state(self, agent, episode):
        if episode % self.save_freq == 0:
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
