from math import prod
import torch
from src.Common.async_single_sim import AsyncSingleSim
from src.Common.common import Common, Logger
from src.REINFORCE.reinforce_agent import ReinforceAgent


# from https://www.youtube.com/watch?v=5eSh5F8gjWU
class ReinforceTrainer:
    def __init__(self, common):
        self.common = common
        self.logger = Logger(common)

        # Reinforce config
        reinforce_config_path = Common.get_file(
            f"./config_files/{common.config.get('reinforce_config_file')}"
        )
        self.reinforce_config = common.load_config_file(reinforce_config_path)
        self.logger.add_file(reinforce_config_path)

        # Configs
        self.device = common.device
        self.debug = common.config.get("debug")
        self.total_timesteps = self.reinforce_config.get("total_timesteps")
        self.discount_factor = self.reinforce_config.get("discount_factor")

        # Sim
        self.sim = AsyncSingleSim(common)

        # Agent
        nn_input_size = prod(self.sim.single_observation_space.shape)
        nn_output_size = self.sim.single_action_space.n
        self.agent = ReinforceAgent(common, nn_input_size, nn_output_size)

        # Train
        self.train()
        self.close()

    def train(self):
        length_episodes = []
        for episode in range(self.total_timesteps):
            losses_len, losses_sum = self.run_episode(episode)
            self.logger.add_scalar("episode_length", losses_len, episode)
            self.logger.add_scalar("losses_sum", losses_sum, episode)
            length_episodes.append(losses_len)
            length_episodes = length_episodes[-15:]
            sum_episodes = sum(length_episodes)
            self.logger.add_scalar("sliding_sum_episodes", sum_episodes, episode)
            self.logger.flush()

        # final reward evaluation
        rewards = []
        for _ in range(5):
            rewards += self.eval_episode()
        self.logger.add_scalar(f"final_reward", sum(rewards), episode)
        self.logger.flush()

    def run_episode(self, episode):
        actions, states, rewards, discounted_returns, losses = [], [], [], [], []
        done = False
        state = self.sim.reset()

        while not done:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            action, log_prob = self.agent.get_action(state)
            next_state, reward, done, info = self.sim.step(action.item())

            if self.debug:
                self.sim.render()

            actions.append(action)
            states.append(state)
            rewards.append(reward)
            state = next_state

        # discounted rewards
        y = self.discount_factor  # discount_factor
        for t in range(len(rewards)):
            g = 0
            for k, r in enumerate(rewards[t:]):
                g += (y**k) * r
            discounted_returns.append(
                torch.tensor(g, dtype=torch.float32, device=self.device)
            )

        for state, action, g in zip(states, actions, discounted_returns):
            _, log_prob = self.agent.get_action(state, action)
            loss = -log_prob * g
            losses.append(loss)
            self.agent.retropropagate(loss)

        print(f"episode {episode} loss ({len(losses)}): {sum(losses).item()}")
        return len(losses), sum(losses)

    def eval_episode(self):
        rewards = []
        done = False
        state = self.sim.reset()

        while not done:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            action = self.agent.get_action(state)
            next_state, reward, done, info = self.sim.step(action.cpu().numpy())

            if self.debug:
                self.sim.render()

            rewards.append(reward)
            state = next_state
        return rewards

    def close(self):
        self.sim.close()
        self.logger.close()
