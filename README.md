# mario_rl
An experiment on reinforcement learning


### Run

Install the environment: `conda env create --file conda.yml`
Activate the environment: `conda activate mario_rl`

Run a training: `python main.py`

### Tensorboard

`tensorboard --logdir runs`

- open: `http://localhost:6006/` in a browser
- remove old logs: `rm -rf logs`