#!/usr/bin/env python
# coding: utf-8
# file: random_maxcut-methods.py
# author: Anthony Wilkie

# %% Import
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt

import networkx as nx


# %% Draw Graph
def draw_cut(G, pos, bitstring, beamer=False):
    S0 = [node for node in G.nodes if bitstring[node] == "0"]
    S1 = [node for node in G.nodes if bitstring[node] == "1"]

    cut_edges = [edge for edge in G.edges if bitstring[edge[0]] != bitstring[edge[1]]]
    uncut_edges = [edge for edge in G.edges if bitstring[edge[0]] == bitstring[edge[1]]]

    nx.draw_networkx_nodes(G, pos, nodelist=S0, node_color='#a99b63')
    #nx.draw_networkx_nodes(G, pos, nodelist=S0, node_color='#936846')
    nx.draw_networkx_nodes(G, pos, nodelist=S1, node_color='#286d8c')
    if beamer:
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='#98c9d3', style='dashdot', alpha=0.5)
        nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, edge_color='#98c9d3', style='solid')
        plt.rc('figure', facecolor='#041017')
    else:
        plt.rcdefaults()
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='#041017', style='dashdot', alpha=0.5)
        nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, edge_color='#041017', style='solid')
    nx.draw_networkx_labels(G, pos)


# %% bitstring
def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)


# %% Plot
barcolors = ['#286d8c', '#a99b63', '#936846', '#4d7888']

def graph(bitstrings, beamer):

    if beamer:
        xticks = range(0, 2**(n_wires))
        xtick_labels = list(map(lambda x: format(x, f"0{n_wires}b"), xticks))
        bins = np.arange(0, 2**(n_wires)+1) - 0.5

        plt.figure(figsize=(16, 8))
        plt.rc('font', size=16)
        plt.rc('axes', edgecolor='#98c9d3', labelcolor='#98c9d3', titlecolor='#98c9d3', facecolor='#041017')
        plt.rc('figure', facecolor='#041017')
        plt.rc('savefig', facecolor='#041017')
        plt.rc('xtick',color='#98c9d3')
        plt.rc('ytick',color='#98c9d3')
        plt.rc('legend',labelcolor='#98c9d3', edgecolor='#98c9d3',facecolor=(0,0,0,0))
        plt.title("s-QAOA")
        plt.xlabel("Bitstrings")
        plt.ylabel("Frequency")
        plt.xticks(xticks, xtick_labels, rotation="vertical")
        plt.hist(bitstrings,
                 bins=bins,
                 density=True,
                 color=barcolors[0],
                 edgecolor = "#041017",
                 # label=[f'scenario {i}' for i in range(n_scenarios)]
                 )
        # plt.legend()
        plt.tight_layout()
        # plt.savefig('/home/vilcius/School/utk/PHYS_642-quantum_information/project/maxcut_1_beamer.pdf',
                   # transparent=True)
    else:
        xticks = range(0, 2**(n_wires))
        xtick_labels = list(map(lambda x: format(x, f"0{n_wires}b"), xticks))
        bins = np.arange(0, 2**(n_wires)+1) - 0.5

        plt.figure(figsize=(16, 8))
        plt.rc('font', size=16)
        plt.title("s-QAOA")
        plt.xlabel("Bitstrings")
        plt.ylabel("Frequency")
        plt.xticks(xticks, xtick_labels, rotation="vertical")
        plt.hist(bitstrings,
                 bins=bins,
                 density=True,
                 color=barcolors[0],
                 edgecolor = "#041017",
                 # label=[f'scenario {i}' for i in range(n_scenarios)]
                 )

        plt.legend()
        plt.tight_layout()
        # plt.savefig('/home/vilcius/School/utk/PHYS_642-quantum_information/project/maxcut_1.pdf',
        #            transparent=True)

    plt.show()


# %% Print Expected Value
def cut_expval(dev, bitstring, C):
    r"""
    Print the expected value of Hamiltonian C with respect to bitstring

    Args:
        dev (data type): TODO
        bitstring (data type): TODO
        C (data type): TODO
    """
    @qml.qnode(dev)
    def _cut_expval(bitstring, C):
        r"""
        Print the expected value of Hamiltonian C with respect to bitstring

        Args:
            bitstring (data type): TODO
            C (data type): TODO
        """
        psi = np.array([int(i) for i in bitstring])
        qml.BasisState(psi,wires=range(len(bitstring)))

        return qml.expval(C)
    return _cut_expval(bitstring, C)


# %% Layer Seeds
def generate_layer_seeds(n_layers, seeds, initial_seed=3, same_seed=True):
    r"""
    Generate the seeds used in each layer of the random circuit.

    Args:
        n_layers (int): the depth of the circuit
        seeds (list): the random seeds to choose from
        initial_seed (int): the initial seed (p=1)
        same_seed (boolean): Whether to use same seed.

    Returns:
        layer_seeds (list): the random seeds used in each layer.
    """
    np.random.seed(initial_seed)
    if n_layers > 1:
        if same_seed:
            layer_seeds = [initial_seed for _ in range(n_layers)]
        else:
            layer_seeds = [initial_seed] + list(np.random.choice(seeds, n_layers-1))
    else:
        layer_seeds = initial_seed
    return layer_seeds

