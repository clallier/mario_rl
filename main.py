from src.Common.common import Common

if __name__ == "__main__":
    common = Common()
    algo = common.config.get("algo")
    trainer = None

    if algo == "NEAT":
        from src.NEAT.neat_trainer import NEATTrainer

        trainer = NEATTrainer(common, algo)

    elif algo == "DQNN":
        from src.DQNN.dqnn_trainer import DQNNTrainer

        trainer = DQNNTrainer(common, algo)

    elif algo == "REINFORCE":
        from src.REINFORCE.reinforce_trainer import ReinforceTrainer

        trainer = ReinforceTrainer(common, algo)

    elif algo == "DPO":
        from src.DPO.dpo_trainer import DPOTrainer

        trainer = DPOTrainer(common, algo)

    if trainer:
        trainer.train()
