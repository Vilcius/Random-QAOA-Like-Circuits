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
        plt.rcdefaults()
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

draw_cut(G, nx.spring_layout(G, seed=1), '1'*n, beamer=False)
plt.axis('off')
plt.savefig('10_vertex_graph.pdf')
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
        qml.Barrier(wires=range(n_wires), only_visual=True)
        for i in range(m):
            wire = np.random.choice(n_wires, size=2, replace=False)
            qml.IsingZZ(gamma[p], wires=[wire.numpy()[0], wire.numpy()[1]])
        qml.Barrier(wires=range(n_wires), only_visual=True)

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
        qml.Barrier(wires=range(n_wires), only_visual=True)
        qml.qaoa.cost_layer(gamma[p], C)
        qml.Barrier(wires=range(n_wires), only_visual=True)
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
        bit_strings.append(bitstring_to_int(circuit(params[0], params[1], seed, sample=True, n_layers=n_layers)))

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    prob = counts[most_freq_bit_string]/n_samples
    print(f"Optimized (gamma, beta) vectors:\n{params[:, :n_layers]}")
    print(f"Most frequently sampled bit string is: {most_freq_bit_string:0{n_wires}b} with probability {prob:.4f}")

    return params, bit_strings, most_freq_bit_string, prob


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
qaoa_dict = {}
p_max = 2

for p in range(1,p_max+1):
    qaoa_dict[p] = {}
    params, bitstrings, most, prob_most = optimize_angles(qaoa_circuit, n_layers=p)
    qaoa_dict[p]['gamma'] = params[0]
    qaoa_dict[p]['beta'] = params[1]
    qaoa_dict[p]['cuts'] = bitstrings
    qaoa_dict[p]['most freq cut'] = most
    qaoa_dict[p]['prob most freq cut'] = prob_most

    ar = []
    for bit in bitstrings:
        ar.append(cut_expval(bit,C)/-13.0)

    prob_dict = {}
    for i in np.sort(ar):
        if i in prob_dict.keys():
            prob_dict[i] += 1
        else:
            prob_dict[i] = 1
    qaoa_dict[p]['AR distribution'] = prob_dict

    qaoa_dict[p]['average AR'] = np.mean(ar)
    qaoa_dict[p]['best AR'] = max(ar)
    qaoa_dict[p]['prob best AR'] = prob_dict[max(ar)]/len(ar)
# graph(bitstrings, beamer=True)

# %% cell name
# print(qaoa_dict[1]['AR distribution'])
print(qaoa_dict[1]['average AR'])
print(qaoa_dict[2]['average AR'])
print(qaoa_dict[1]['prob best AR'])
print(qaoa_dict[2]['prob best AR'])
print(qaoa_dict[1]['prob most freq cut'])
print(qaoa_dict[2]['prob most freq cut'])

# %% Generate Multiple Random Circuits
num_circs = 50

random_circuits_same = {}

# set random seed to generate random seeds
np.random.seed(1)
seeds = np.random.choice(10000, num_circs)
p_max = 2

for p in range(1,p_max+1):
    random_circuits_same[p] = {}
    i_gamma = []
    i_beta = []
    i_bitstrings = []
    i_most = []
    i_prob_most = []
    i_ar = []
    i_avg_ar = []
    i_best_ar = []
    i_prob_best_ar = []
    for i in range(num_circs):
        print('------------------------------------------------------------')
        print(f"Random Circuit #{i+1}, p = {p}")
        layer_seeds = generate_layer_seeds(n_layers=p, seeds=seeds, initial_seed=seeds[i], same_seed=True)
        params, bitstrings, most, prob_most = optimize_angles(random_circuit, seed=layer_seeds, n_layers=p)
        i_gamma.append(params[0])
        i_beta.append(params[1])
        i_bitstrings.append(bitstrings)
        i_most.append(most)
        i_prob_most.append(prob_most)

        ar = []
        for bit in bitstrings:
            ar.append(cut_expval(bit,C)/-13.0)

        prob_dict = {}
        for ii in np.sort(ar):
            if ii in prob_dict.keys():
                prob_dict[ii] += 1
            else:
                prob_dict[ii] = 1
        i_ar.append(prob_dict)

        i_avg_ar.append(np.mean(ar))
        i_best_ar.append(max(ar))
        i_prob_best_ar.append(prob_dict[max(ar)]/len(ar))

    random_circuits_same[p]['gamma'] = i_gamma
    random_circuits_same[p]['beta'] = i_beta
    random_circuits_same[p]['cuts'] = i_bitstrings
    random_circuits_same[p]['most freq cut'] = i_most
    random_circuits_same[p]['prob most freq cut'] = i_prob_most
    random_circuits_same[p]['AR distribution'] = i_ar
    random_circuits_same[p]['average AR'] = i_avg_ar
    random_circuits_same[p]['best AR'] = i_best_ar
    random_circuits_same[p]['prob best AR'] = i_prob_best_ar


# %% Generate Multiple Random Circuits
num_circs = 50

random_circuits_diff = {}

# set random seed to generate random seeds
np.random.seed(1)
seeds = np.random.choice(10000, num_circs)
p_max = 2

for p in range(1,p_max+1):
    random_circuits_diff[p] = {}
    i_gamma = []
    i_beta = []
    i_bitstrings = []
    i_most = []
    i_prob_most = []
    i_ar = []
    i_avg_ar = []
    i_best_ar = []
    i_prob_best_ar = []
    for i in range(num_circs):
        print('------------------------------------------------------------')
        print(f"Random Circuit #{i+1}, p = {p}")
        layer_seeds = generate_layer_seeds(n_layers=p, seeds=seeds, initial_seed=seeds[i], same_seed=False)
        params, bitstrings, most, prob_most = optimize_angles(random_circuit, seed=layer_seeds, n_layers=p)
        i_gamma.append(params[0])
        i_beta.append(params[1])
        i_bitstrings.append(bitstrings)
        i_most.append(most)
        i_prob_most.append(prob_most)

        ar = []
        for bit in bitstrings:
            ar.append(cut_expval(bit,C)/-13.0)

        prob_dict = {}
        for ii in np.sort(ar):
            if ii in prob_dict.keys():
                prob_dict[ii] += 1
            else:
                prob_dict[ii] = 1
        i_ar.append(prob_dict)

        i_avg_ar.append(np.mean(ar))
        i_best_ar.append(max(ar))
        i_prob_best_ar.append(prob_dict[max(ar)]/len(ar))

    random_circuits_diff[p]['gamma'] = i_gamma
    random_circuits_diff[p]['beta'] = i_beta
    random_circuits_diff[p]['cuts'] = i_bitstrings
    random_circuits_diff[p]['most freq cut'] = i_most
    random_circuits_diff[p]['prob most freq cut'] = i_prob_most
    random_circuits_diff[p]['AR distribution'] = i_ar
    random_circuits_diff[p]['average AR'] = i_avg_ar
    random_circuits_diff[p]['best AR'] = i_best_ar
    random_circuits_diff[p]['prob best AR'] = i_prob_best_ar

# %% cell name
print(np.max(random_circuits_same[1]['AR distribution']))

# %% Show MaxCut
# draw_cut(G, nx.spring_layout(G, seed=1), f'{most_freq_cut_rand:0{n_wires}b}', True)
# plt.axis('off')
# plt.show()

# graph(bitstrings_rand, beamer=True)

# %% Show MaxCut QAOA
draw_cut(G, nx.spring_layout(G, seed=1), f'{qaoa_dict[1]["most freq cut"]:0{n_wires}b}', False)
plt.axis('off')
plt.savefig('maxcut_10_vertex.pdf')
plt.show()
print(cut_expval(qaoa_dict[1]["most freq cut"],C)/-13.0)

# graph(bitstrings_qaoa, beamer=True)


# %% test
most_freq_cut_qaoa
print(most_freq_cut_qaoa in random_most)

i=0
for most in random_most_same[1]:
    i = i + 1
    print('------------------------------------------------------------')
    print(f'QAOA #{i}:   {cut_expval(most_freq_cut_qaoa,C)/-13.0}')
    print(f'Random #{i}: {cut_expval(most,C)/-13.0}')


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
qaoa_fig, qaoa_ax = qml.draw_mpl(qaoa_circuit, decimals=1)(qaoa_dict[1]['gamma'], qaoa_dict[1]['beta'], n_layers=1)
qaoa_fig.suptitle("QAOA Circuit", fontsize="xx-large")
plt.savefig('paper/qaoa_circuit_1.pdf')

# rand_max_fig, rand_max_ax = qml.draw_mpl(random_circuit, style='default', decimals=2)(random_params[15][0], random_params[15][1], seeds[15])
# rand_max_fig.suptitle("Max AR Random Circuit", fontsize="xx-large")
# plt.savefig('max_random_circuit.pdf')

# rand_min_fig, rand_min_ax = qml.draw_mpl(random_circuit, style='default', decimals=2)(random_params[5][0], random_params[5][1], seeds[5])
# rand_min_fig.suptitle("Min AR Random Circuit", fontsize="xx-large")
# plt.savefig('min_random_circuit.pdf')


# %% testtttt
draw_cut(G, nx.spring_layout(G, seed=1), f'{most_freq_cut_qaoa:0{n_wires}b}', True)
plt.savefig('maxcut_10_vertex.pdf')
# draw_cut(G, nx.spring_layout(G, seed=1), f'{random_most[15]:0{n_wires}b}', True). 
# draw_cut(G, nx.spring_layout(G, seed=1), f'{random_most[5]:0{n_wires}b}', True)
# plt.axis('off')
# plt.show()


# %% Write Data
def write_data(circuit, p, angles, bitstrings, most_freq, best_ar, avg_ar,  seed_type=None):
    r"""
    Documention of method

    Args:
        circuit (data type): TODO
    p (data type): TODO
    angles (data type): TODO
    bitstrings (data type): TODO
    most_freq (data type): TODO
    ar (data type): TODO
    seed_type (data type): TODO
=None (data type): TODO

    Returns:
        return value
    """
    # Create file name
    if seed_type==None:
        file = f'{circuit}_p={p}.csv'
    else:
        file = f'{circuit}_p={p}_seed={seed_type}.csv'

    with open(file, 'w') as f:
        # Write header
        f.write('Circuit,')
        f.write('p,')
        if seed_type != None:
            f.write('seed type,')
        f.write('Average AR,')
        f.write('Best AR,')
        for i in range(p):
            f.write(f'gamma_{i+1},')
        for i in range(p):
            if i == p-1:
                f.write(f'beta_{i+1}\n')
            else:
                f.write(f'beta_{i+1},')

        # Write data
        if seed_type == None:
            f.write(f'{circuit},')
            f.write(f'{p},')
            f.write(f'{avg_ar},')
            f.write(f'{best_ar},')
            for i in range(p):
                f.write(f'{angles[0][i]},')
            for i in range(p):
                if i == p-1:
                    f.write(f'{angles[1][i]}')
                else:
                    f.write(f'{angles[1][i]},')
        else:
            for c in range(len(avg_ar)):
                f.write(f'{circuit}_{c+1},')
                f.write(f'{p},')
                f.write(f'{seed_type}')
                f.write(f'{avg_ar[c]},')
                f.write(f'{best_ar[c]},')
                for i in range(p):
                    f.write(f'{angles[c][0][i]},')
                for i in range(p):
                    if i == p-1:
                        if c == len(avg_ar)-1:
                            f.write(f'{angles[c][1][i]}')
                        else:
                            f.write(f'{angles[c][1][i]}\n')
                    else:
                        f.write(f'{angles[c][1][i]},')

    if seed_type==None:
        file = f'{circuit}_p={p}_bits.csv'
    else:
        file = f'{circuit}_p={p}_seed={seed_type}_bits.csv'

    with open(file, 'w') as f:




for p in range(1,p_max+1):
    write_data(circuit='QAOA', p=p, angles=qaoa_params[p], bitstrings=qaoa_bitstrings[p], most_freq=qaoa_most[p], avg_ar=qaoa_avg_ar[p], best_ar=qaoa_best_ar[p], seed_type=None)

# for p in range(1,3):
#     write_data(circuit='random', p=p, angles=random_params_same[p], bitstrings=random_bitstrings_same[p], most_freq=random_most_same[p], avg_ar=random_ar, best_ar=qaoa_ar_best, seed_type=None)




# %% t
prob_dict = {}
for i in np.sort(qaoa_avg_ar[1]):
    if i in prob_dict.keys():
        prob_dict[i] += 1
    else:
        prob_dict[i] = 1
print(prob_dict)
print(prob_dict[max(prob_dict.keys())]/len(qaoa_avg_ar[1]))

# %% file
with open('QAOA_bits.txt','w') as f:
    f.write(str(qaoa_bitstrings))



# %% stuff
print('------------------------------------------------------------')
print(f"{qaoa_dict[2]['average AR']:.6f}")
print('------------------------------------------------------------')
print(f"{np.average(random_circuits_same[2]['average AR']):.6f}")
print(f"{np.min(random_circuits_same[2]['average AR']):.6f}")
print(f"{np.max(random_circuits_same[2]['average AR']):.6f}")
print('------------------------------------------------------------')
print(f"{np.average(random_circuits_diff[2]['average AR']):.6f}")
print(f"{np.min(random_circuits_diff[2]['average AR']):.6f}")
print(f"{np.max(random_circuits_diff[2]['average AR']):.6f}")


p=2
same_index = random_circuits_same[p]['average AR'].index(np.max(random_circuits_diff[p]['average AR']))
diff_index = random_circuits_diff[p]['average AR'].index(np.max(random_circuits_diff[p]['average AR']))

# qaoa_fig, qaoa_ax = qml.draw_mpl(qaoa_circuit, decimals=2)(qaoa_dict[p]['gamma'], qaoa_dict[p]['beta'], n_layers=p)
# qaoa_fig.suptitle(f"QAOA Circuit, p={p}", fontsize="xx-large")
# plt.savefig(f'paper/qaoa_circuit_{p}.pdf')

# layer_seeds = generate_layer_seeds(n_layers=p, seeds=seeds, initial_seed=seeds[same_index], same_seed=True)
# rand_max_fig, rand_max_ax = qml.draw_mpl(random_circuit, decimals=2)(random_circuits_same[p]['gamma'][same_index], random_circuits_same[p]['beta'][same_index], layer_seeds,n_layers=p)
# rand_max_fig.suptitle(f"Random QAOA-like circuit with maximum AR, p={p}", fontsize="xx-large")
# plt.savefig(f'paper/random_circuit_max_ar_{p}.pdf')

layer_seeds = generate_layer_seeds(n_layers=p, seeds=seeds, initial_seed=seeds[diff_index], same_seed=False)
rand_diff_max_fig, rand_diff_max_ax = qml.draw_mpl(random_circuit, decimals=2)(random_circuits_diff[p]['gamma'][diff_index], random_circuits_diff[p]['beta'][diff_index], layer_seeds,n_layers=p)
rand_diff_max_fig.suptitle(f"Random QAOA-like circuit with maximum AR, p={p}, different", fontsize="xx-large")
plt.savefig(f'paper/random_circuit_max_ar_{p}_diff.pdf')

# %% cell name
p=2
layer_seeds = generate_layer_seeds(n_layers=p, seeds=seeds, initial_seed=seeds[diff_index], same_seed=False)
print(layer_seeds)
