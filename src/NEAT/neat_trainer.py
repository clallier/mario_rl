from pathlib import Path

from src.Common.sim import Sim
from src.Common.common import Common, Logger
import neat

from src.NEAT.neat_agent import NeatAgent
from src.NEAT.neat_statistics_logger import StatisticsLogger


# from Tech with Tim:
# https://www.youtube.com/watch?v=wQWWzBHUJWM

class NEATTrainer:
    def __init__(self):
        self.common = Common()
        self.logger = Logger(True)

        config_file_path = Path(Path(__file__).parent, 'config.v3.cfg')
        self.logger.add_file(config_file_path)

        self.config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file_path)
        self.train()

    def train(self):
        p = neat.Population(self.config)
        stats_logger = StatisticsLogger(self.logger)
        p.add_reporter(stats_logger)
        p.run(self.eval_genomes, 1000)
        return stats_logger.save()

    def test(self, genome):
        sim = Sim(self.common)
        agent = NeatAgent(genome, self.config, sim, True)

        while not agent.done:
            action = agent.choose_action(agent.state)
            next_state, reward, done, info = agent.sim.step(action)
            # agent.update_fitness(next_state, reward, done, info)

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
                NeatAgent(genome[1],
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
