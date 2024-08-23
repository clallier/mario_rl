from src.Common.common import Common

if __name__ == "__main__":
    common = Common()

    algo = common.config.get('algo')

    if algo == 'NEAT':
        from src.NEAT.neat_trainer import NEATTrainer
        NEATTrainer(common)

    elif algo == 'DQNN':
        from src.DQNN.dqnn_trainer import DQNNTrainer
        DQNNTrainer(common, 0)

    elif algo == 'REINFORCE':
        from src.REINFORCE.reinforce_trainer import ReinforceTrainer
        ReinforceTrainer(common)

    elif algo == 'DPO':
        from src.DPO.dpo_trainer import DPOTrainer
        DPOTrainer(common)
