from src.Common.common import Common, Logger
from src.Common.sim import Sim
from src.REINFORCE.reinforce_agent import ReinforceAgent


# from https://www.youtube.com/watch?v=5eSh5F8gjWU
class ReinforceTrainer:
    def __init__(self, common):
        self.common = common
        self.logger = Logger(common)
        sim = Sim(common)

        self.agent = ReinforceAgent(common, sim)
        self.train()
        self.close()

    def train(self):
        length_episodes = []
        for step in range(10_000):
            n, losses = self.agent.run_episode(step)
            self.logger.add_scalar("episode_length", n, step)
            self.logger.add_scalar("losses_sum", losses, step)
            length_episodes.append(n)
            length_episodes = length_episodes[-15:]
            sum_episodes = sum(length_episodes)
            self.logger.add_scalar('sum_episodes', sum_episodes, step)
            self.logger.flush()
            if sum_episodes > 15 * 485:
                break

        # final reward evaluation
        rewards = []
        for _ in range(5):
            rewards += self.agent.eval_episode()
        self.logger.add_scalar(f"final_reward (/{5 * 499})", sum(rewards), step)
        self.logger.flush()

    def close(self):
        self.agent.close()
        self.logger.close()
