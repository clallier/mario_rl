from math import prod
from pathlib import Path
import torch
from src.Common.async_single_sim import AsyncSingleSim
from src.Common.trainer import Trainer
from src.REINFORCE.reinforce_agent import ReinforceAgent
from src.Common.conv_calc import debug_count_params, debug_nn_size


# from https://www.youtube.com/watch?v=5eSh5F8gjWU
class ReinforceTrainer(Trainer):
    def init(self):
        self.discount_factor = self.config.get("discount_factor")
        if self.debug:
            state = self.sim.reset()
            debug_nn_size(self.agent.nn, state, self.device)
            debug_count_params(self.agent.nn)

    def create_sim(self):
        return AsyncSingleSim(self.common)

    def create_agent(self):
        nn_input_size = prod(self.sim.single_observation_space.shape)
        nn_output_size = self.sim.single_action_space.n
        return ReinforceAgent(self.common, nn_input_size, nn_output_size, self.config)

    def train_init(self):
        self.length_episodes = []

    # def train(self):
    #     for episode in range(self.num_episodes):
    #         self.run_episode(episode)

    #     # final reward evaluation
    #     rewards = []
    #     for _ in range(5):
    #         rewards += self.eval_episode()
    #     self.logger.add_scalar(f"final_reward", sum(rewards), episode)
    #     self.logger.flush()
    #     self.close

    def run_episode(self, episode):
        actions, states, rewards, discounted_returns, losses = [], [], [], [], []
        done = False
        state = self.sim.reset()

        while not done:
            state = self.to_tensor(state)
            action, log_prob = self.agent.get_action(state)
            next_state, reward, done, info = self.sim.step(action.item())

            if self.debug:
                self.sim.render()

            actions.append(action)
            states.append(state)
            rewards.append(info.get("normalized_reward"))
            state = next_state

        # discounted rewards
        y = self.discount_factor  # discount_factor
        for t in range(len(rewards)):
            g = 0
            for k, r in enumerate(rewards[t:]):
                g += (y**k) * r
            discounted_returns.append(self.to_tensor(g))

        for state, action, g in zip(states, actions, discounted_returns):
            _, log_prob = self.agent.get_action(state, action)
            loss = -log_prob * g
            losses.append(loss)
            self.agent.retropropagate(loss)

        # log some data
        losses_len = len(losses)
        losses_sum = sum(losses).item()
        print(f"episode {episode} losses_len: {losses_len}, losses_sum: {losses_sum}")
        self.logger.add_scalar("episode_length", losses_len, episode)
        self.logger.add_scalar("losses_sum", losses_sum, episode)

        self.length_episodes.append(losses_len)
        self.length_episodes = self.length_episodes[-15:]
        sum_episodes = sum(self.length_episodes)
        self.logger.add_scalar("sliding_sum_episodes", sum_episodes, episode)

    def eval_episode(self):
        rewards = []
        done = False
        state = self.sim.reset()

        while not done:
            state = self.to_tensor(state)
            action = self.agent.get_action(state)
            next_state, reward, done, info = self.sim.step(action.cpu().numpy())

            if self.debug:
                self.sim.render()

            rewards.append(reward)
            state = next_state
        return rewards

    def save_complete_state(self, path: Path):
        torch.save(
            {
                "episode": self.episode,
                "optimizer": self.agent.optimizer.state_dict(),
                "scheduler": self.agent.scheduler.state_dict(),
                "neural_network": self.agent.nn.state_dict(),
                "sim": self.sim.state_dict(),
            },
            path,
        )

    def load_complete_state(self, path):
        load_state = torch.load(path, weights_only=False)
        self.episode = load_state["episode"]
        self.agent.optimizer.load_state_dict(load_state["optimizer"])
        self.agent.scheduler.load_state_dict(load_state["scheduler"])
        model_state_dict = load_state["neural_network"]
        self.agent.nn.load_state_dict(model_state_dict)
        self.sim.load_state_dict(load_state["sim"])
