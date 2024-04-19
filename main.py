from src.Common.common import Common
from src.DQNN.dqnn_trainer import DQNNTrainer
from src.NEAT.neat_trainer import NEATTrainer

if __name__ == "__main__":
    common_config = Common.load_config_file("main_config.toml")
    if common_config['algo'] == 'NEAT':
        NEATTrainer(common_config)
    else:
        DQNNTrainer(0, common_config)
