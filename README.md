# mario_rl
An experiment on reinforcement learning


### Run

Install the environment: `conda env create --file ./mario_rl/environment.yml`
Activate the environment: `conda activate mario_rl_py3.11`

Run a training: `python ./mario_rl/main.py`

### Tensorboard

`tensorboard --logdir runs`

- open: `http://localhost:6006/` in a browser
- remove old logs: `rm -rf logs`