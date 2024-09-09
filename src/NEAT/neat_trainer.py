from pathlib import Path
from src.Common.sim import Sim
from src.Common.common import background
import neat

from src.Common.trainer import Trainer
from src.NEAT.neat_agent import NeatAgent
from src.NEAT.neat_statistics_logger import StatisticsLogger
import asyncio


# from Tech with Tim:
# https://www.youtube.com/watch?v=wQWWzBHUJWM

# TODO:
# - wandb: more logs
# - store population every n steps


class NEATTrainer(Trainer):
    def create_sim(self):
        pass

    def create_agent(self):
        pass

    def init(self):
        self.neat_config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path,
        )

    def train_init():
        return super.train_init()

    def run_episode() -> dict:
        return super.run_episode()

    def train(self):
        gen_num = int(self.config.get("META", {}).get("gen_num", 100))
        p = neat.Population(self.neat_config)
        self.stats_logger = StatisticsLogger(self.logger)
        p.add_reporter(self.stats_logger)
        p.run(self.eval_genomes, n=gen_num)
        return self.stats_logger.save()

    def test(self, genome):
        sim = Sim(self.common)
        agent = NeatAgent(genome, self.neat_config, sim, True)

        while not agent.done:
            action = agent.choose_action(agent.state)
            agent.sim.step(action)

    def eval_genomes(self, genomes, config):
        """
        Compute the fitness of the genomes by running the game with the genomes and return the fitness of each genome.
        :param genomes:
        :param config:
        :return:
        """
        print("creating agents pop")
        loop = asyncio.get_event_loop()
        looper = asyncio.gather(
            *[self.new_agent(config, genome, i) for i, genome in enumerate(genomes)]
        )
        agents = loop.run_until_complete(looper)

        # run the sim step by step for each agent
        # until all agents are done
        print("running agents pop")
        while any([not agent.done for agent in agents]):
            # loop = asyncio.get_event_loop()
            looper = asyncio.gather(
                *[self.eval_agent(agents[i]) for i in range(len(agents))]
            )
            loop.run_until_complete(looper)

        sorted_agents = sorted(agents, key=lambda a: a.genome.fitness)
        best_agent = sorted_agents[-1]
        generation = self.stats_logger.generation
        print(f"End of gen: {generation}, best fitness: {best_agent.genome.fitness}")
        self.end_of_episode(best_agent.info, generation)

    def eval_genomes_no_parallel(self, genomes, config):
        """
        Compute the fitness of the genomes by running the game with the genomes and return the fitness of each genome.
        :param genomes:
        :param config:
        :return:
        """
        agents = []
        # create agent for each genome
        for i, genome in enumerate(genomes):
            agents.append(self.new_agent(config, genome, i))

        # run the sim step by step for each agent
        # until all agents are done
        while any([not agent.done for agent in agents]):
            for agent in agents:
                self.eval_agent(agent)

    @background
    def new_agent(self, config, genome, i):
        if config == None or genome == None:
            return

        return NeatAgent(
            genome[1], config, Sim(self.common), self.common.debug and i == 0
        )

    @background
    def eval_agent(self, agent: NeatAgent):
        if not agent.done:
            action = agent.choose_action(agent.state)
            next_state, reward, done, info = agent.sim.step(action)
            agent.update_fitness(next_state, reward, done, info)

    def save_complete_state(self, path: Path):
        pass

    def load_complete_state(self, path: Path):
        pass
