from pathlib import Path
import time

import numpy as np
import torchvision as torchvision
from neat import DefaultGenome

from src.Common.common import Logger
from src.NEAT import visualize

import neat
from neat.six_util import itervalues, iterkeys


class StatisticsLogger(neat.StatisticsReporter):
    def __init__(self, start_tensorboard=True):
        super().__init__()
        self.num_extinctions = 0
        self.generation_times = []
        self.generation = 0
        self.generation_start_time = 0
        self.flag_get_sum = 0
        self.logger = Logger(start_tensorboard)

    def info(self, msg):
        print(msg)

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')

    def start_generation(self, generation):
        super().start_generation(generation)
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    # This is call before end_generation
    def post_evaluate(self, config, population, species, best_genome: DefaultGenome):
        super().post_evaluate(config, population, species, best_genome)
        self.logger.add_scalar("Best genome fitness", best_genome.fitness, self.generation)
        # size = (number of nodes, number of enabled connections)
        size = best_genome.size()
        self.logger.add_scalar("Best genome num nodes", size[0], self.generation)
        self.logger.add_scalar("Best genome num connections", size[1], self.generation)
        self.logger.add_scalar("Best genome key", best_genome.key, self.generation)

        # flag_get
        self.flag_get_sum += sum([c.flag_get for c in itervalues(population)])
        self.logger.add_scalar('flag_get_sum', self.flag_get_sum, self.generation)

        # Store the fitness's of the members of each currently active species.
        fitnesses = np.array([c.fitness for c in itervalues(population)])
        fit_mean = fitnesses.mean()
        fit_std = fitnesses.std()
        best_species_id = species.get_species_id(best_genome.key)
        self.logger.add_histogram('Finesses', fitnesses, self.generation)
        print(f'Population\'s average fitness: {fit_mean:3.5f} stdev: {fit_std:3.5f}')
        print(
            f'Best fitness: {best_genome.fitness:3.5f} - size: {best_genome.size()!r} - species {best_species_id} - id {best_genome.key}')

    def end_generation(self, config, population, species_set):
        print('\n ****** End generation {0} ****** \n'.format(self.generation))

        super().end_generation(config, population, species_set)
        ng = len(population)
        ns = len(species_set.species)
        print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
        self.logger.add_scalar('Population', ng, self.generation)
        self.logger.add_scalar('Species', ns, self.generation)

        sids = list(iterkeys(species_set.species))
        sids.sort()
        print("   ID   age  size  fitness  adj fit  stag")
        print("  ====  ===  ====  =======  =======  ====")
        for sid in sids:
            s = species_set.species[sid]
            a = self.generation - s.created
            n = len(s.members)
            f = "--" if s.fitness is None else f"{s.fitness:.1f}"
            af = "--" if s.adjusted_fitness is None else f"{s.adjusted_fitness:.3f}"
            st = self.generation - s.last_improved
            print(
                "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        self.logger.add_scalar('Generation Time', elapsed, self.generation)
        average = sum(self.generation_times) / len(self.generation_times)
        if len(self.generation_times) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))

        # TODO
        print('Total extinctions: {0:d}'.format(self.num_extinctions))
        self.logger.add_scalar('Extinctions', self.num_extinctions, self.generation)
        self.logger.flush()

    def save(self):
        # We need to overload the behaviour of the StatisticsReporter.save()
        # super().save()
        # TODO
        #   1 - save best
        #   3 - compute genome distances
        #   4 - add/remove nodes
        #   5 - plot_stats (avg_fitness.svg) -> tsboard
        #   6 - plot_species (speciation.svg) -> tsboard
        #   7 - config dict -> tsboard
        winner = self.best_genome()

        self.save_species_count()
        self.save_species_fitness()

        visualize.plot_stats(self)
        visualize.plot_species(self)
        # plot the best of the last generation
        for prog in ['sfdp', 'twopi', 'neato']:
            name = f'best_genome_{prog}'
            filename = Path(self.logger.log_dir, f'{name}.svg')
            visualize.draw_net(filename, winner, prog)

        self.logger.close()
