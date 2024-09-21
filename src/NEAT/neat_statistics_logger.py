import math
import time

import numpy as np
from neat import DefaultGenome
from neat.species import GenomeDistanceCache

from src.NEAT import visualize
import neat


class StatisticsLogger(neat.StatisticsReporter):
    def __init__(self, common, trainer, logger):
        super().__init__()
        self.num_extinctions = 0
        self.generation_times = []
        self.generation = 0
        self.generation_start_time = 0
        self.flag_get_sum = 0
        self.logger = logger
        self.common = common
        self.trainer = trainer
        self.debug = common.debug

    def info(self, msg):
        print(msg)

    def complete_extinction(self):
        self.num_extinctions += 1
        print("All species extinct.")

    def start_generation(self, generation):
        super().start_generation(generation)
        self.generation = generation
        print(f"\n ****** Running generation {generation} ****** \n")
        self.generation_start_time = time.time()

    # This is call before end_generation
    def post_evaluate(self, config, population, species, best: DefaultGenome):
        super().post_evaluate(config, population, species, best)
        self.logger.add_scalar("Best genome fitness", best.fitness, self.generation)
        # size = (number of nodes, number of enabled connections)
        size = best.size()
        self.logger.add_scalar("Best genome num nodes", size[0], self.generation)
        self.logger.add_scalar("Best genome num connections", size[1], self.generation)
        self.logger.add_scalar("Best genome key", best.key, self.generation)

        # flag_get
        self.flag_get_sum += sum([c.info.get("flag_get") for c in population.values()])
        self.logger.add_scalar("flag_get_sum", self.flag_get_sum, self.generation)

        # Store the fitness's of the members of each currently active species.
        pop_fitnesses = np.array([c.fitness for c in population.values()])
        pop_fitnesses_mean = pop_fitnesses.mean()
        pop_fitnesses_std = pop_fitnesses.std()
        print(f"Pop average fit: {pop_fitnesses_mean:.2f} std: {pop_fitnesses_std:.2f}")

        try:
            best_species_id = species.get_species_id(best.key)
            print(
                f"Best fit: {best.fitness:.2f}, size: {best.size()!r}, species {best_species_id}, id {best.key}"
            )
        except:
            print(
                f"WARNING: error with species.get_species_id(best.key), best.key: {best.key}"
            )

        self.logger.add_histogram("Finesses", pop_fitnesses, self.generation)
        self.logger.add_scalar("Pop fitness avg", pop_fitnesses_mean, self.generation)
        self.logger.add_scalar("Pop fitness std", pop_fitnesses_std, self.generation)

    def end_generation(self, config, population, species_set):
        super().end_generation(config, population, species_set)
        print(f"\n ****** End generation {self.generation} ****** \n")

        pop_len = len(population)
        species_count = len(species_set.species)
        print(f"Population of {pop_len:d} members in {species_count:d} species")
        self.logger.add_scalar("Population", pop_len, self.generation)
        self.logger.add_scalar("Species", species_count, self.generation)
        species_keys = sorted(species_set.species.keys())

        if self.debug:
            print("   ID   age  size  fitness  adj fit  stag")
            print("  ====  ===  ====  =======  =======  ====")
            for key in species_keys:
                current = species_set.species[key]
                age = self.generation - current.created
                size = len(current.members)
                fitness = "--" if current.fitness is None else f"{current.fitness:.1f}"
                adjusted_fitness = (
                    "--"
                    if current.adjusted_fitness is None
                    else f"{current.adjusted_fitness:.3f}"
                )
                stagnation = self.generation - current.last_improved
                print(
                    f"  {key: >4}  {age: >3}  {size: >4}  {fitness: >7}  {adjusted_fitness: >7}  {stagnation: >4}"
                )

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        self.logger.add_scalar("Generation Time", elapsed, self.generation)
        average = sum(self.generation_times) / len(self.generation_times)
        if len(self.generation_times) > 1:
            print(f"Generation time: {elapsed:.3f} sec ({average:.3f} average)")
        else:
            print(f"Generation time: {elapsed:.3f} sec")

        print(f"Total extinctions: {self.num_extinctions:d}")
        self.logger.add_scalar("Extinctions", self.num_extinctions, self.generation)

        # Get genetic dist to the best representatives for each species
        keys = population.keys()
        distances = GenomeDistanceCache(config.genome_config)
        for key in species_keys:
            current = species_set.species[key]
            for key in keys:
                g = population[key]
                distances(current.representative, g)

        dists = np.array(list(distances.distances.values()))
        dist_mean = dists.mean()
        dist_std = dists.std()
        print(f"> Genetic distance, mean: {dist_mean:3.5f}, std: {dist_std:3.5f}")
        self.logger.add_scalar("Mean genetic distance", dist_mean, self.generation)
        self.logger.add_scalar("Std genetic distance", dist_std, self.generation)

        best_genome = self.best_genome()
        self.trainer.end_of_episode(best_genome.info, self.generation)

    def save(self):
        # We need to overload the behaviour of the StatisticsReporter.save()
        # super().save()
        winner = self.best_genome()
        self.logger.add_pickle("best_genome", winner)

        self.logger.add_figure("avg_fitness", visualize.plot_stats(self))
        self.logger.add_figure("speciation", visualize.plot_species(self))
        # plot the best of the last generation
        for prog in ["sfdp", "twopi", "neato"]:
            name = f"best_genome_{prog}"
            self.logger.add_figure(name, visualize.draw_net(winner, prog))

        self.logger.close()
        return winner
