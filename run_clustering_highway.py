import os
import numpy as np
import argparse
from utils.utils import estimate_distribution_params, filter_grads
import time
import pickle
from algorithms.clustering import em_clustering

# Directories where the agent policies, trajectories and gradients (if already calcualted) are stored
# To add agents populate this dictionary and store the gradients in '/gradients.npy'
# Or if u want to calculate the gradients directly store the policy as a tf checkpoint in a file called best
# and the trajectories in the subfolder 'trajectories/<subfolder>/K_trajectories.csv'

feature_labels = np.array(['r_freeright', 'r_lanechange', 'r_distancefront'])


agent_to_data = {
        "Alice": ["logs/tensorboards/highway/Alice/50_1_8_1583075761.0390627", []],
        "Chuck": ["logs/tensorboards/highway/Chuck/50_1_8_1583076699.3430083", []],
        "Carol": ["logs/tensorboards/highway/Carol/50_1_8_1583075749.288957", []],
        "Bob": ["logs/tensorboards/highway/Bob/50_1_8_1583075749.2889063", []],
        "Dan": ["logs/tensorboards/highway/Dan/50_1_8_1583156123.1914694", []],
        "Craig": ["logs/tensorboards/highway/Craig/50_1_8_1583079229.988111", []],
        "Eve": ["logs/tensorboards/highway/Eve/50_1_8_1583159023.0972116", []],
        "Grace": ["logs/tensorboards/highway/Grace/50_1_8_1583168762.0848062", []],
        "Erin": ["logs/tensorboards/highway/Erin/50_1_8_1583156364.5193923", []],
        "Judy": ["logs/tensorboards/highway/Judy/50_1_8_1583076699.3434803", []],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_idx', default='', type=str, help='indexes of the features to perform irl on')
    parser.add_argument('--verbose', action='store_true', help='print logs on terminal')
    parser.add_argument('--ep_len', type=int, default=400, help='length of trajectories')
    parser.add_argument('--gamma', type=float, default=0.9999, help='discount_factor')
    parser.add_argument('--num_clusters', type=int, default=4, help='number of clusters')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--max_iterations', type=int, default=100, help='maximum em iterations')
    parser.add_argument('--cluster_iterations', type=int, default=5, help='cluster iterations')
    parser.add_argument('--opt_iters', type=int, default=5, help='iterations of optimization of sigma girl loss')
    parser.add_argument('--lamb', type=float, default=0.5, help='lambda parameter of em clustering')
    args = parser.parse_args()

    EPISODE_LENGTH = args.ep_len
    if args.features_idx == '':
        features_idx = [0, 1, 2]
    else:
        features_idx = [int(x) for x in args.features_idx.split(',')]
    num_objectives = len(features_idx)
    res = np.zeros((9, 20, 3))
    mus = []
    sigmas = []
    ids = []

    for i, agent in enumerate(agent_to_data.keys()):
        read_path = agent_to_data[agent][0]
        estimated_gradients = np.load(read_path + '/gradients.npy', allow_pickle=True)
        estimated_gradients = estimated_gradients[:, :, features_idx]
        estimated_gradients = filter_grads(estimated_gradients, verbose=args.verbose)
        num_episodes, num_parameters, num_objectives = estimated_gradients.shape[:]
        mu, sigma = estimate_distribution_params(estimated_gradients=estimated_gradients,
                                                 diag=False, identity=False,
                                                 cov_estimation=True, other_options=[False, False],
                                                 girl=False)
        id_matrix = np.identity(num_parameters)
        mus.append(mu)
        sigmas.append(sigma)
        ids.append(id_matrix)
    np.random.seed(args.seed)

    losses = []
    Ps = []
    Omegas = []
    for i in range(args.cluster_iterations):
        print("Clustering Iteration %d" % (i+1))
        P, Omega, loss = em_clustering(mus, sigmas, ids, num_clusters=args.num_clusters,
                                       num_objectives=num_objectives,
                                       max_iterations=args.max_iterations, verbose=args.verbose,
                                       optimization_iterations=args.opt_iters,
                                       lamb=args.lamb)
        Ps.append(P)
        Omegas.append(Omega)
        losses.append(loss)
    loss = np.min(losses)
    best_index = np.argmin(losses)
    P = Ps[best_index]
    Omega = Omegas[best_index]
    print("Did EM clustering 10 times variance of losses: " + str(np.std(losses)))
    print(losses)

    results = {}
    out_path = 'logs/highway/clustering/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    feature_labels = feature_labels[features_idx]
    agent_names = list(agent_to_data.keys())
    results["agents"] = agent_names
    results["assignment"] = P
    results["weights"] = Omega
    results["loss"] = loss
    results["features"] = feature_labels
    timestamp = str(time.time())

    pickle.dump(results, open(out_path + 'results_' + timestamp, 'wb'))
    print("Agents")
    print(agent_names)
    print("Features")
    print(feature_labels)
    print("Assignment:")
    print(P)
    print("Weights:")
    print(Omega)
    print("Loss:")
    print(loss)
