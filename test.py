# Adapted from Sourish Kundu's video : https://www.youtube.com/watch?v=_gmQZToTMac

import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace

from common import Common
from src.wrapper import apply_wrappers
from src.DQNN.dqnn_agent import DQNNAgent
from pathlib import Path

def test_from_checkpoint(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    print("### loading checkpoint from: ", checkpoint_path.resolve(), checkpoint_path.exists())

    common = Common(start_logger=False)
    env = gym_super_mario_bros.make(Common.ENV_NAME)
    env.metadata['render.modes'] = ['rgb_array']
    env.metadata['apply_api_compatibility'] = True

    env = JoypadSpace(env, Common.RIGHT_RUN)
    env = apply_wrappers(env)
    agent = DQNNAgent(env.observation_space.shape, env.action_space.n, common)
    agent.load_state(checkpoint_path)

    done = False
    state = env.reset()
    reward_sum_raw = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        reward_sum_raw += reward["raw"]
        state = next_state
        env.render()
    print("### reward_sum: ", reward_sum_raw)
    env.close()
    return reward_sum_raw

if __name__ == "__main__":
    checkpoint_path = "runs/Feb15_22-47-59_Corentins-MacBook-Pro.local/checkpoints/agent_checkpoint_0.pt"
    scores = []
    for i in range(10):
        scores.append(test_from_checkpoint(checkpoint_path))
    print("### scores: ", scores, "mean: ", np.mean(scores), "stdev: ", np.std(scores), "best: ", np.max(scores), "worst: ", np.min(scores))
