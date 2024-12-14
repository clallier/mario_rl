import numpy as np
import torch
import time

from pathlib import Path
from skvideo.io import vwrite
from src.Common.async_single_sim import AsyncSingleSim
from src.Common.common import Common


def test_from_actions_path(action_pt_path: str, common: Common):
    path = Path(action_pt_path)
    print("### loading actions from: ", path.resolve(), path.exists())
    actions = torch.load(path)
    # e.g: 
    # path ="../../runs/Feb15_22-47-59/actions/agent_actions_ep:6985_rw:3058.pt"
    test_from_actions(actions, common, path)


def test_from_actions(actions: [], common: Common, output_video_path=None):
    sim = AsyncSingleSim(common)
    images = []

    for action in actions:
        _, _, _, _ = sim.step(action)

        if output_video_path:
            rgb_array = sim.env.render(mode="rgb_array")
            copy_array = np.array(rgb_array, dtype=np.uint8)
            images.append(copy_array)
        else:
            sim.render()

    time.sleep(3)
    sim.close()

    if output_video_path:
        images = np.array(images, dtype=np.uint8)
        print("### images.shape: ", images.shape)
        video_path = Path(output_video_path.parent, f"{output_video_path.stem}.mp4")

        # scikit video
        vwrite(video_path, images)
        print("### video saved at: ", video_path)
