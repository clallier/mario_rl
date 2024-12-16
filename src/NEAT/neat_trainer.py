from pathlib import Path
from pprint import pp
from src.Common.async_multi_sim import AsyncMultiSims
from src.Common.run_from_actions import test_from_actions
from src.Common.sim import Sim
from src.Common.common import background
import neat

from src.Common.trainer import Trainer
from src.NEAT.neat_agent import NeatAgent
from src.NEAT.neat_statistics_logger import StatisticsLogger
import asyncio
import pickle


# from Tech with Tim:
# https://www.youtube.com/watch?v=wQWWzBHUJWM


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
        self.population = neat.Population(self.neat_config)
        self.stats_logger = StatisticsLogger(self.common, self, self.logger)
        self.population.add_reporter(self.stats_logger)

    def train_init(self):
        pass

    def train(self):
        gen_num = int(self.config.get("META", {}).get("gen_num", 100))
        self.population.run(self.run_episode, n=gen_num)

        return self.stats_logger.save()

    def test(self, genome):
        sim = Sim(self.common)
        agent = NeatAgent(genome, self.neat_config, sim, True)

        while not agent.done:
            action = agent.choose_action(agent.prev_state)
            agent.env.step(action)

    def run_episode(self, genomes, config) -> dict:
        """
        Compute the fitness of the genomes by running the game with the genomes and return the fitness of each genome.
        :param genomes:
        :param config:
        :return:
        """
        len_genomes = len(genomes)
        self.episode = self.population.generation

        print(f"creating agents pop {len_genomes}")
        # TODO: create_sim?
        self.sim = AsyncMultiSims(self.common, len(genomes))
        states = self.sim.reset()

        loop = asyncio.get_event_loop()
        looper = asyncio.gather(
            *[
                self.new_agent(config, genome, states[i])
                for i, genome in enumerate(genomes)
            ]
        )
        agents = loop.run_until_complete(looper)

        # run the sim step by step for each agent
        # until all agents are done
        print("running agents pop")
        while any([not agent.done for agent in agents]):
            # loop = asyncio.get_event_loop()
            looper = asyncio.gather(
                *[self.eval_agent(agents[i], i) for i in range(len(agents))]
            )
            loop.run_until_complete(looper)

        self.sim.close()

        if self.episode % 10 == 0:
            i = agents.index(max(agents, key=lambda a: a.genome.fitness))
            g = agents[i].genome
            print(f"### best, id:{g.key}, fit: {g.fitness}, ")
            pp(g.info, width=120, compact=True)
            # self.sim.envs[i].render()
            actions = g.info.get("episode").get("a")
            test_from_actions(actions, self.common)

    @background
    def new_agent(self, config, genome, init_state):
        if config is None or genome is None:
            return
        return NeatAgent(genome[1], config, init_state)

    @background
    def eval_agent(self, agent: NeatAgent, i: int):
        if not agent.done:
            next_info = self.info_to_tensor(agent.genome.info)
            action = agent.choose_action(agent.prev_state, next_info)
            state, _, done, info = self.sim.envs[i].step(action)
            agent.update_fitness(state, done, info)

    def save_complete_state(self, path: Path):
        state = (
            self.population.population,
            self.population.species.genome_to_species,
            self.population.species.species,
            self.population.generation,
            self.population.best_genome,
        )
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_complete_state(self, path: Path):
        with open(path, "rb") as f:
            state = pickle.load(f)
            (
                self.population.population,
                self.population.species.genome_to_species,
                self.population.species.species,
                self.population.generation,
                self.population.best_genome,
            ) = state
            self.population.generation += 1
