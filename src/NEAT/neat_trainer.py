from pathlib import Path

from src.Common.sim import Sim
from src.Common.common import Common
import neat

from src.NEAT.neat_agent import NeatAgent
from src.NEAT.neat_statistics_logger import StatisticsLogger


# from Tech with Tim:
# https://www.youtube.com/watch?v=wQWWzBHUJWM

class NEATTrainer:
    def __init__(self):
        self.common = Common()
        config_file_path = Path(Path(__file__).parent, 'config.cfg')

        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file_path)

        p = neat.Population(config)
        logger = StatisticsLogger(True)
        p.add_reporter(logger)

        p.run(self.eval_genomes, 100)
        logger.save()

    def eval_genomes(self, genomes, config):
        """
        Compute the fitness of the genomes by running the game with the genomes and return the fitness of each genome.
        :param genomes:
        :param config:
        :return:
        """
        agents = []
        # create agent for each genome
        for i, genome in enumerate(genomes):
            agents.append(
                NeatAgent(genome,
                          config,
                          Sim(self.common),
                          self.common.debug and i == 0)
            )
            
        # run the sim step by step for each agent
        # until all agents are done        
        while any([not agent.done for agent in agents]):
            for agent in agents:
                if not agent.done:
                    action = agent.choose_action(agent.state)
                    next_state, reward, done, info = agent.sim.step(action)
                    agent.update_fitness(next_state, reward, done, info)
