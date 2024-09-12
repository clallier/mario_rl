import neat
import numpy as np


class NeatAgent:
    def __init__(self, genome, config, init_state):
        self.prev_state = init_state
        self.done = False
        self.info = {}

        self.genome_key = genome.key
        self.genome = genome
        self.genome.fitness = 0
        self.genome.info = {}
        self.config = config
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, config)

    def choose_action(self, state):
        # flatten the state
        state = np.array(state).flatten()
        output = self.net.activate(state)
        # print(f"output: {output}")
        # output to action
        output = output.index(max(output))
        return output

    def update_fitness(self, state, done, info):
        self.prev_state = state
        if "episode" in info.keys():
            self.genome.fitness = info["episode"]["r"].item()
            self.genome.info = info
            self.done = done
