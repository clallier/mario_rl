import torch
from torch.distributions import Categorical
from torch.nn import Sequential, Linear, ReLU, Softmax
from torch.optim import Adam


class ReinforceAgent:

    def __init__(self, common, sim):
        self.common = common
        self.device = common.device
        self.sim = sim
        self.nn = Sequential(
            Linear(4, 64),
            ReLU(),
            Linear(64, sim.env.single_action_space.n),
            Softmax(dim=-1)
        )
        self.nn.to(self.device)

        lr = 0.001
        self.optim = Adam(self.nn.parameters(), lr=lr)

    def run_episode(self, i, debug=False):
        actions, states, rewards, discounted_returns, losses = [], [], [], [], []
        done = False
        prev_state = self.sim.reset()

        while not done:
            prev_state = torch.tensor(prev_state, dtype=torch.float32, device=self.device)
            probs = self.nn(prev_state)
            dist = Categorical(probs)
            action = dist.sample()
            action_numpy = action.cpu().numpy()
            state, reward, done, _ = self.sim.env.step(action_numpy)

            if debug:
                self.sim.env.render()

            actions.append(action)
            states.append(prev_state)
            rewards.append(reward)
            prev_state = state

        # discounted rewards
        y = 0.99  # discount_factor
        for t in range(len(rewards)):
            g = 0
            for k, r in enumerate(rewards[t:]):
                g += (y ** k) * r
            discounted_returns.append(torch.tensor(g, dtype=torch.float32, device=self.device))

        for state, action, g in zip(states, actions, discounted_returns):
            probs = self.nn(state)
            dist = Categorical(probs)
            log_prob = dist.log_prob(action)
            loss = -log_prob * g
            losses.append(loss)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        print(f'episode {i} loss ({len(losses)}): {sum(losses)}')
        return len(losses), sum(losses)

    def eval_episode(self, debug=False):
        rewards = []
        done = False
        prev_state = self.sim.reset()

        while not done:
            prev_state = torch.tensor(prev_state, dtype=torch.float32, device=self.device)
            probs = self.nn(prev_state)
            dist = Categorical(probs)
            action = dist.sample()
            action_numpy = action.cpu().numpy()
            state, reward, done, _ = self.sim.env.step(action_numpy)

            if debug:
                self.sim.env.render()

            rewards.append(reward)
            prev_state = state
        return rewards

    def close(self):
        self.sim.close()
