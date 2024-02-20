import gym_super_mario_bros
import numpy as np
import torch
from nes_py.wrappers import JoypadSpace

from common import Common
from src.wrapper import apply_wrappers

from pathlib import Path
from skvideo.io import vwrite

def test_from_actions(action_pt_path:str):
    path = Path(action_pt_path)
    print("### loading actions from: ", path.resolve(), path.exists())
    actions = torch.load(path)
    
    env = gym_super_mario_bros.make(Common.ENV_NAME)
    env.metadata['render.modes'] = ['rgb_array']
    env.metadata['apply_api_compatibility'] = True

    env = JoypadSpace(env, Common.RIGHT_RUN)
    env = apply_wrappers(env)

    state = env.reset()
    reward_raw_sum = 0
    coins = 0
    final_score = 0
    images = []
    print("### env reset, state: ", state)
    len_actions = len(actions)

    for i, action in enumerate(actions):
        next_state, reward, done, info = env.step(action)
        reward_raw_sum += reward['raw']
        coins += info['coins']
        final_score += info['score']
        print("i: ", i, "/", len_actions,
              "action: ", action,
              "rewards: ", reward_raw_sum,
              "coins: ", coins,
              "xpos: ", info['x_pos'],
              "flag_get: ", info['flag_get'])
        # env.render()
        rgb_array = env.render(mode='rgb_array')
        # copy the array to avoid references    
        images.append(np.array(rgb_array, dtype=np.uint8))
    print("### reward_sum: ", reward_raw_sum)
    env.close()

    images = np.array(images, dtype=np.uint8)
    print("### images.shape: ", images.shape)
    video_path = Path(path.parent, f"{path.stem}.mp4")

    # scikit video
    vwrite(video_path, images)
    print("### video saved at: ", video_path)


if __name__ == "__main__":
    test_from_actions("runs/Feb15_22-47-59_Corentins-MacBook-Pro.local/actions/agent_actions_ep:6985_rw:3058.pt")
