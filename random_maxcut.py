#!/usr/bin/env python
# coding: utf-8
# file: random_maxcut.py
# author: Anthony Wilkie

# TODO: Find more efficient way to calculate expected value (do not create new circuit)
# TODO: Implement better data post-processing (pandas)
# TODO: Implement option to do ma-QAOA


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
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='#041017', style='dashdot', alpha=0.5)
        nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, edge_color='#041017', style='solid')
    nx.draw_networkx_labels(G, pos)


# %% Create Random Graph
n = 10
p = 0.3
graph_seed = 18
G = nx.gnp_random_graph(n, p, seed=graph_seed)
m = len(G.edges())

# n = 5
# edges = [(0,1), (0,2), (0,3), (0,4), (1,2), (2,3)]
# m = len(edges)
# G = nx.Graph(edges)

draw_cut(G, nx.spring_layout(G, seed=1), '1'*n, beamer=True)
plt.axis('off')
plt.show()


# %% bitstring
def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)


# %% Variables
n_wires = n
dev = qml.device("default.qubit", wires=n, shots=1)


# %% MaxCut Hamiltonian
C, B = qml.qaoa.maxcut(G)
print(C)


# %% Circuit
@qml.qnode(dev)
def random_circuit(gamma, beta, seed=12345, sample=False, probs=False, n_layers=1):
    if n_layers == 1:
        seed = [seed]

    # initialize state to be an equal superposition
    for i in range(n_wires):
        qml.Hadamard(i)

    for p in range(n_layers):
        # choose m random gates for the cost Hamiltonian
        np.random.seed(seed[p])
        for i in range(m):
            wire = np.random.choice(n_wires, size=2, replace=False)
            qml.IsingZZ(gamma[p], wires=[wire.numpy()[0], wire.numpy()[1]])

        # Apply mixing Hamiltionian
        qml.qaoa.mixer_layer(beta[p], B)

    # return samples instead of expectation value
    if sample:
        # measurement phase
        return qml.sample()

    # return probabilities instead of expectation value
    if probs:
        return qml.probs(wires=range(n_wires))

    # Currently use the sum of the Cost Hamiltonians.
    return qml.expval(C)


# %% QAOA Circuit
@qml.qnode(dev)
def qaoa_circuit(gamma, beta, seed=None, sample=False, probs=False, n_layers=1):
    # initialize state to be an equal superposition
    for i in range(n_wires):
        qml.Hadamard(i)

    # apply unitary layers
    for p in range(n_layers):
        qml.qaoa.cost_layer(gamma[p], C)
        qml.qaoa.mixer_layer(beta[p], B)

    # return samples instead of expectation value
    if sample:
        # measurement phase
        return qml.sample()

    # return probabilities instead of expectation value
    if probs:
        return qml.probs(wires=range(n_wires))

    # Currently use the sum of the Cost Hamiltonians.
    return qml.expval(C)


# %% Optimization
# np.random.seed(1248)

def optimize_angles(circuit, seed=None, n_layers=1):
    print("\np={:d}".format(n_layers))

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

    # minimize the negative of the objective function
    def objective(params):
        gammas = params[0]
        betas = params[1]
        neg_obj = circuit(gammas, betas, seed, n_layers=n_layers)
        return neg_obj

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # optimize parameters in objective
    params = init_params
    steps = 50
    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))

    # sample measured bitstrings 1000 times
    bit_strings = []
    n_samples = 1000
    for i in range(0, n_samples):
        bit_strings.append(bitstring_to_int(circuit(params[0], params[1], sample=True, n_layers=n_layers)))

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print(f"Optimized (gamma, beta) vectors:\n{params[:, :n_layers]}")
    print(f"Most frequently sampled bit string is: {most_freq_bit_string:0{n_wires}b} with probability {(counts[most_freq_bit_string]/n_samples):.4f}")

    return params, bit_strings, most_freq_bit_string


# %% Plot
import matplotlib.pyplot as plt
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
@qml.qnode(dev)
def cut_expval(bitstring, C):
    r"""
    Print the expected value of Hamiltonian C with respect to bitstring

    Args:
        bitstring (data type): TODO
        C (data type): TODO
    """
    psi = np.array([int(i) for i in list(f'{bitstring:0{n}b}')])
    qml.BasisState(psi,wires=range(n_wires))

    return qml.expval(C)


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


# %% Run the QAOA Thing
params_qaoa, bitstrings_qaoa, most_freq_cut_qaoa = optimize_angles(qaoa_circuit, n_layers=1)
# graph(bitstrings, beamer=True)



# %% Generate Multiple Random Circuits
num_circs = 50
random_params = {}
random_bitstrings = {}
random_most = {}
random_expvals = {}

# set random seed to generate random seeds
np.random.seed(1)
seeds = np.random.choice(10000, num_circs)
p_max = 2

for p in range(p_max):
    i_params = []
    i_bitstrings = []
    i_most = []
    i_expvals = []
    for i in range(num_circs):
        print('------------------------------------------------------------')
        print(f"Random Circuit #{i+1}")
        layer_seeds = generate_layer_seeds(n_layers=p+1, seeds=seeds, initial_seed=seeds[i], same_seed=True)
        params, bitstrings, most = optimize_angles(random_circuit, seed=layer_seeds, n_layers=p+1)
        i_params.append(params)
        i_bitstrings.append(bitstrings)
        i_most.append(most)
        i_expvals.append(random_circuit(params[0], params[1]))

    random_params[p+1] = i_params
    random_bitstrings[p+1] = i_bitstrings
    random_most[p+1] = i_most
    random_expvals[p+1] = i_expvals


# %% test
# print(qml.draw_mpl(random_circuit, style='solarized_dark', decimals=2)(params_rand[0], params_rand[1]))
print(qml.draw_mpl(qaoa_circuit, style='solarized_dark', decimals=2)(params_qaoa[0], params_qaoa[1]))

# %% Show MaxCut
# draw_cut(G, nx.spring_layout(G, seed=1), f'{most_freq_cut_rand:0{n_wires}b}', True)
# plt.axis('off')
# plt.show()

# graph(bitstrings_rand, beamer=True)

# %% Show MaxCut QAOA
draw_cut(G, nx.spring_layout(G, seed=1), f'{most_freq_cut_qaoa:0{n_wires}b}', True)
plt.axis('off')
plt.show()

# graph(bitstrings_qaoa, beamer=True)


# %% test
most_freq_cut_qaoa
print(most_freq_cut_qaoa in random_most)

i=0
for most in random_most:
    i = i + 1
    print('------------------------------------------------------------')
    print(f'QAOA #{i}:   {cut_expval(most_freq_cut_qaoa,C)/-13.0}')
    print(f'Random #{i}: {cut_expval(most,C)/-13.0}')


# %% Calculate Average AR
qaoa_ar = 0
random_ar = []

for bit in bitstrings_qaoa:
    qaoa_ar = qaoa_ar + cut_expval(bit,C)/-13.0
qaoa_ar = qaoa_ar/len(bitstrings_qaoa)

for bits in random_bitstrings:
    ar = 0
    for bit in bits:
        ar = ar + cut_expval(bit,C)/-13.0
    random_ar.append(ar/len(bits))

# %% test
for i in range(num_circs):
    print('------------------------------------------------------------')
    print(f'QAOA Average AR with QAOA optimal angles:             {qaoa_ar:.4f}')
    print(f'Random #{i+1} Average AR with Random #{i+1} optimal angles: {random_ar[i]:.4f}')

# %% testt
print(f'random random AR range: [{min(random_ar):.4f}, {max(random_ar):.4f}]')
print(f'qaoa random AR range: [{min(qaoa_random_ar):.4f}, {max(qaoa_random_ar):.4f}]')

# %% testtt
print(f'QAOA Average AR:                 {qaoa_ar:.4f}')
print(f'average of all random random AR: {np.average(random_ar):.4f}')


# %% testttt
print(np.argmax(random_ar), random_ar[15])
print(np.argmin(random_ar), random_ar[5])


qaoa_fig, qaoa_ax = qml.draw_mpl(qaoa_circuit, style='default', decimals=2)(params_qaoa[0], params_qaoa[1])
qaoa_fig.suptitle("QAOA Circuit", fontsize="xx-large")
plt.savefig('qaoa_circuit.pdf')

rand_max_fig, rand_max_ax = qml.draw_mpl(random_circuit, style='default', decimals=2)(random_params[15][0], random_params[15][1], seeds[15])
rand_max_fig.suptitle("Max AR Random Circuit", fontsize="xx-large")
plt.savefig('max_random_circuit.pdf')

rand_min_fig, rand_min_ax = qml.draw_mpl(random_circuit, style='default', decimals=2)(random_params[5][0], random_params[5][1], seeds[5])
rand_min_fig.suptitle("Min AR Random Circuit", fontsize="xx-large")
plt.savefig('min_random_circuit.pdf')


# %% testtttt
# draw_cut(G, nx.spring_layout(G, seed=1), f'{most_freq_cut_qaoa:0{n_wires}b}', True)
draw_cut(G, nx.spring_layout(G, seed=1), f'{random_most[15]:0{n_wires}b}', True). 
# draw_cut(G, nx.spring_layout(G, seed=1), f'{random_most[5]:0{n_wires}b}', True)
# plt.axis('off')
# plt.show()




