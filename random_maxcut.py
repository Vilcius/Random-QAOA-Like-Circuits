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
import random_maxcut-methods as mthd


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

mthd.draw_cut(G, nx.spring_layout(G, seed=1), '1'*n, beamer=True)
plt.axis('off')
plt.savefig('10_vertex_graph.pdf')
plt.show()


# %% Variables
n_wires = n
shots=1000
dev = qml.device("default.qubit", wires=n, shots=shots)


# %% MaxCut Hamiltonian
C, B = qml.qaoa.maxcut(G)
print(C)


# %% Random Circuit
@qml.qnode(dev)
def random_circuit(gamma, beta, seed=12345, counts=False, probs=False, n_layers=1):
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
    if counts:
        # measurement phase
        return qml.counts(all_coutcomes=True)

    # return probabilities instead of expectation value
    if probs:
        return qml.probs(wires=range(n_wires))

    # Currently use the sum of the Cost Hamiltonians.
    return qml.expval(C)


# %% QAOA Circuit
@qml.qnode(dev)
def qaoa_circuit(gamma, beta, seed=None, counts=False, probs=False, n_layers=1):
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
    if counts:
        # measurement phase
        return qml.counts(all_outcomes=True)

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
    n_counts = shots
    counts = circuit(params[0], params[1], seed, counts=True, n_layers=n_layers)

    # print optimal parameters and most frequently sampled bitstring
    most_freq_bit_string = max(counts, key=counts.get)
    prob = counts[most_freq_bit_string]/n_samples
    print(f"Optimized (gamma, beta) vectors:\n{params[:, :n_layers]}")
    print(f"Most frequently sampled bit string is: {most_freq_bit_string} with probability {prob:.4f}")

    return params, bit_strings, most_freq_bit_string, prob


# %% Run the QAOA Thing
qaoa_dict = {}
p_max = 1

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
        layer_seeds = mthd.generate_layer_seeds(n_layers=p, seeds=seeds, initial_seed=seeds[i], same_seed=True)
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
        layer_seeds = mthd.generate_layer_seeds(n_layers=p, seeds=seeds, initial_seed=seeds[i], same_seed=False)
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
mthd.draw_cut(G, nx.spring_layout(G, seed=1), f'{qaoa_dict[1]["most freq cut"]:0{n_wires}b}', False)
plt.axis('off')
plt.savefig('maxcut_10_vertex.pdf')
plt.show()
print(mthd.cut_expval(qaoa_dict[1]["most freq cut"],C)/-13.0)

# graph(bitstrings_qaoa, beamer=True)


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
mthd.draw_cut(G, nx.spring_layout(G, seed=1), f'{most_freq_cut_qaoa:0{n_wires}b}', True)
plt.savefig('maxcut_10_vertex.pdf')
# draw_cut(G, nx.spring_layout(G, seed=1), f'{random_most[15]:0{n_wires}b}', True). 
# draw_cut(G, nx.spring_layout(G, seed=1), f'{random_most[5]:0{n_wires}b}', True)
# plt.axis('off')
# plt.show()



for p in range(1,p_max+1):
    mthd.write_data(circuit='QAOA', p=p, angles=qaoa_params[p], bitstrings=qaoa_bitstrings[p], most_freq=qaoa_most[p], avg_ar=qaoa_avg_ar[p], best_ar=qaoa_best_ar[p], seed_type=None)

# for p in range(1,3):
#     write_data(circuit='random', p=p, angles=random_params_same[p], bitstrings=random_bitstrings_same[p], most_freq=random_most_same[p], avg_ar=random_ar, best_ar=qaoa_ar_best, seed_type=None)


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

layer_seeds = mthd.generate_layer_seeds(n_layers=p, seeds=seeds, initial_seed=seeds[diff_index], same_seed=False)
rand_diff_max_fig, rand_diff_max_ax = qml.draw_mpl(random_circuit, decimals=2)(random_circuits_diff[p]['gamma'][diff_index], random_circuits_diff[p]['beta'][diff_index], layer_seeds,n_layers=p)
rand_diff_max_fig.suptitle(f"Random QAOA-like circuit with maximum AR, p={p}, different", fontsize="xx-large")
plt.savefig(f'paper/random_circuit_max_ar_{p}_diff.pdf')
