import warnings

import pygraphviz
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()

def debug_draw_net():
    g = pygraphviz.AGraph(directed=True, format='svg')
    w = 30
    h = 30 * 4
    x = y = 0

    for i in range(0, -3600, -3):
        node_attr = {}
        if i < 0:
            x = i % w
            y = -i // w
        print(f"node {i}, x: {x}, y: {y}")
        node_attr['pos'] = f"{x}, {y}!"
        g.add_node(i, node_attr=node_attr)

    # top tier: neato, twopi
    # ok tier: fdp, sfdp, circo
    g.draw("best_genome.svg", prog="neato")


def draw_net(filename, genome, prog='sfdp'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if pygraphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    node_attrs = {
        'shape': 'circle',
        'fontsize': '8',
        'color': '#00000022'
    }

    g = pygraphviz.AGraph(directed=True, format='svg', node_attrs=node_attrs)

    for cg in genome.connections.values():
        src, dst = cg.key
        style = 'solid' if cg.enabled else 'dotted'
        color = '#ff00ffaa' if cg.weight > 0 else '#00ffffaa'
        width = str(0.1 + abs(cg.weight * 2.0))
        g.add_edge(src, dst, style=style, color=color, penwidth=width)

    w = 30
    h = 30 * 4
    x = y = 0
    for node in g.nodes():
        i = int(node.name)
        if i < 0:
            x = i % w
            y = -i // w
            node.attr['color'] = '#55555555'
        elif i == 0:
            x = w + 4
            y = h // 2
            node.attr['root'] = True
            node.attr['shape'] = 'box'
        else:
            x = w + 2
            y = i % h
            print(f"node {i}, x: {x}, y: {y}")
        node.attr['pos'] = f"{x}, {y}!"

    # top tier: neato, twopi
    # ok tier: fdp, sfdp, circo
    g.draw(filename, prog=prog)

    return g
