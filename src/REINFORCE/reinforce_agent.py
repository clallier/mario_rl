from torch.distributions import Categorical
from torch import nn
from torch.optim import Adam


class ReinforceAgent:

    def __init__(self, common, nn_input_size, nn_output_size):
        self.common = common
        self.device = common.device

        hidden_size = nn_input_size // 2
        hidden_size_2 = hidden_size // 2
        hidden_size_4 = hidden_size_2 // 2
        self.nn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nn_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_4),
            nn.ReLU(),
            nn.Linear(hidden_size_4, nn_output_size),
            nn.Softmax(dim=-1),
        )
        self.nn.to(self.device)

        lr = 0.001
        self.optimizer = Adam(self.nn.parameters(), lr=lr)

    def get_action(self, x, action=None):
        probs = self.nn(x)
        dist = Categorical(probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def retropropagate(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
