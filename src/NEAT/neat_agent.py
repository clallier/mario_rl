import neat
import numpy as np


class NeatAgent:
    def __init__(self, genome, config, sim, debug=False):
        self.debug = debug
        self.sim = sim
        self.state = sim.reset()
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
        if self.debug:
            print(f"output: {output}")
        # output (relu activation) to binary action
        output = 1 if output[0] > 0.5 else 0
        return output

    def update_fitness(self, next_state, reward, done, info):
        if "episode" in info.keys():
            self.genome.fitness = info["episode"]["r"].item()
            self.genome.info = info
            self.done = done

        self.state = next_state
        # if self.debug:
        #     self.debug_loop()
        if self.done:
            self.sim.close()

    def debug_loop(self):
        try:
            self.sim.env.render()
        finally:
            print(self.done, self.genome.fitness, self.info)
