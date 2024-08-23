import neat
import numpy as np


class NeatAgent:
    def __init__(self, genome, config, sim, debug=False):
        self.debug = debug
        self.sim = sim
        self.state = sim.reset()
        self.fitness_raw = 0
        self.done = False
        self.info = {}

        self.genome_key = genome.key
        self.genome = genome
        self.genome.fitness = 0
        self.genome.flag_get = False
        self.config = config
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, config)

    def choose_action(self, state):
        # flatten the state
        state = np.array(state).flatten()
        output = self.net.activate(state)
        if self.debug:
            print(f"output: {output}")
        # output (relu activation) to binary action
        output = 1 if output[0] > 0.5 else 0
        return output

    def update_fitness(self, next_state, reward, done, info):
        self.fitness_raw += info['normalized_reward']
        self.genome.fitness += info['reward']
        self.genome.flag_get = info['flag_get']
        self.state = next_state
        self.done = done
        self.info = info
        if self.debug:
            self.debug_loop()
        if self.done:
            self.sim.close()

    def debug_loop(self):
        try:
            self.sim.env.render()
        finally:
            print(self.done, self.genome.fitness, self.info)
