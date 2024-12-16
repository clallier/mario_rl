import warnings

import pygraphviz
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def plot_stats(statistics, ylog=False):
    """Plots the population's average and best fitness."""
    if plt is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (matplotlib)"
        )
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, "b-", label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, "g-.", label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, "g-.", label="+1 sd")
    plt.plot(generation, best_fitness, "r-", label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale("symlog")

    figure = plt.gcf()
    plt.close()
    return figure


def plot_species(statistics):
    """Visualizes speciation throughout evolution."""
    if plt is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (matplotlib)"
        )
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    figure = plt.gcf()
    plt.close()
    return figure


def draw_net(genome, prog="sfdp"):
    """Receives a genome and draws a neural network with arbitrary topology."""
    # Attributes for network nodes.
    if pygraphviz is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (graphviz)"
        )
        return

    node_attrs = {"shape": "circle", "fontsize": "8", "color": "#00000022"}

    g = pygraphviz.AGraph(directed=True, format="svg", node_attrs=node_attrs)

    for cg in genome.connections.values():
        src, dst = cg.key
        style = "solid" if cg.enabled else "dotted"
        color = "#ff00ff" if cg.weight > 0 else "#00ffff"
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
            node.attr["color"] = "#55555555"
        elif i == 0:
            x = w + 4
            y = h // 2
            node.attr["root"] = True
            node.attr["shape"] = "box"
        else:
            x = w + 2
            y = i % h
            print(f"node {i}, x: {x}, y: {y}")
        node.attr["pos"] = f"{x}, {y}!"

    # top tier: neato, twopi
    # ok tier: fdp, sfdp, circo
    g.layout(prog)
    return draw_using_networkx(g)


def draw_using_networkx(g):
    # draw the graph g using matplotlib and networkx
    # first need to convert the graph to a networkx graph
    nx_g = nx.nx_agraph.from_agraph(g)
    # get the nodes' positions (convert from string to float)
    pos = [node.attr["pos"].split(",") for node in g.nodes()]
    pos = {node: (float(x), -float(y)) for node, (x, y) in zip(g.nodes(), pos)}

    # get nodes color
    node_color = [node.attr["color"] for node in g.nodes()]
    node_color = [c if c != "" else "#00000022" for c in node_color]

    # draw the nodes
    nx.draw_networkx_nodes(nx_g, pos, node_color=node_color)

    # get edges color and width
    edge_colors = [e.attr["color"] for e in g.edges()]
    edge_width = [2.0 * float(e.attr["penwidth"]) for e in g.edges()]

    # draw the edges
    nx.draw_networkx_edges(
        nx_g,
        pos,
        arrowstyle="->",
        arrowsize=5,
        edge_color=edge_colors,
        alpha=0.7,
        width=edge_width,
    )

    # draw the labels
    nx.draw_networkx_labels(nx_g, pos, font_size=7, alpha=0.8)

    figure = plt.gcf()
    plt.close()
    return figure
