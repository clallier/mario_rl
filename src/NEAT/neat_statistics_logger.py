import time

import numpy as np
from neat import DefaultGenome
from neat.species import GenomeDistanceCache

from src.Common.sim import Sim
from src.NEAT import visualize
import neat


class StatisticsLogger(neat.StatisticsReporter):
    def __init__(self, logger):
        super().__init__()
        self.num_extinctions = 0
        self.generation_times = []
        self.generation = 0
        self.generation_start_time = 0
        self.flag_get_sum = 0
        self.logger = logger

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
        self.flag_get_sum += sum([c.flag_get for c in population.values()])
        self.logger.add_scalar('flag_get_sum', self.flag_get_sum, self.generation)

        # Store the fitness's of the members of each currently active species.
        pop_fitness = np.array([c.fitness for c in population.values()])
        fit_mean = pop_fitness.mean()
        fit_std = pop_fitness.std()
        best_species_id = species.get_species_id(best_genome.key)
        self.logger.add_histogram('Finesses', pop_fitness, self.generation)
        print(f'Pop\'s average fit: {fit_mean:3.5f} std: {fit_std:3.5f}')
        print(f'Best fit: {best_genome.fitness:3.5f} - size: {best_genome.size()!r}'
              f' - species {best_species_id} - id {best_genome.key}')
        self.logger.add_scalar("Pop fitness avg", fit_mean, self.generation)
        self.logger.add_scalar("Pop fitness std", fit_std, self.generation)

    def end_generation(self, config, population, species_set):
        super().end_generation(config, population, species_set)

        print('\n ****** End generation {0} ****** \n'.format(self.generation))

        ng = len(population)
        ns = len(species_set.species)
        print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
        self.logger.add_scalar('Population', ng, self.generation)
        self.logger.add_scalar('Species', ns, self.generation)

        sids = sorted(species_set.species.keys())
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

        print('Total extinctions: {0:d}'.format(self.num_extinctions))
        self.logger.add_scalar('Extinctions', self.num_extinctions, self.generation)

        # Get genetic dist to the best representatives for each species
        gids = population.keys()
        distances = GenomeDistanceCache(config.genome_config)
        sids = sorted(species_set.species.keys())
        for sid in sids:
            s = species_set.species[sid]
            for gid in gids:
                g = population[gid]
                distances(s.representative, g)

        dists = np.array(list(distances.distances.values()))
        gdmean = dists.mean()
        gdstdev = dists.std()
        print(f'>>>> Mean genetic distance {gdmean:3.5f}, standard deviation {gdstdev:3.5f}')
        self.logger.add_scalar("Mean genetic distance", gdmean, self.generation)
        self.logger.add_scalar("Std genetic distance", gdstdev, self.generation)

        self.logger.flush()

    def save(self):
        # We need to overload the behaviour of the StatisticsReporter.save()
        # super().save()
        winner = self.best_genome()
        self.logger.add_pickle("best_genome", winner)

        self.logger.add_figure("avg_fitness", visualize.plot_stats(self))
        self.logger.add_figure("speciation", visualize.plot_species(self))
        # plot the best of the last generation
        for prog in ['sfdp', 'twopi', 'neato']:
            name = f'best_genome_{prog}'
            self.logger.add_figure(name, visualize.draw_net(winner, prog))

        self.logger.close()
        return winner

