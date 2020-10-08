import os
import sys
import numpy as np
import argparse
import csv
import time
path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path
from utils.utils import compute_gradient
from algorithms.sigma_girl import solve_sigma_PGIRL
from bc.behaviornal_clonning_twitter_clustering import bc_twitter
import pickle


def log(message):
    if args.verbose:
        print(message)


parser = argparse.ArgumentParser()
parser.add_argument("--dir_to_read", type=str, default='', help='directory to read the datasets')
parser.add_argument('--settings', default='', type=str, help='irl algorithms to run')
parser.add_argument('--horizon', type=int, default=20, help='horizon of the task')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--seed', type=float, default=1, help='random seed')
parser.add_argument('--num_experiments', type=int, default=1, help='number of experiments to average upon')
parser.add_argument('--verbose', action='store_true', help='enable_logging')
parser.add_argument('--save_gradients', action='store_true', help='save the computed gradients')
parser.add_argument('--load_gradients', action='store_true', help='load the precomputed policy')

#Following parameters get passed to the bc_twitter method
parser.add_argument('--validation', type=float, default=0.1, help='size of validation set')
parser.add_argument('--stochastic_eval', action='store_true', help='evaluate accuracy with stochastic policy')
parser.add_argument('--save_best', action='store_true', help='save the best policy')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--l2', type=float, default=0., help='l2 regularization hyperparameter')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--ratio', type=int, default=1.1, help='threshhold on action imbalance to perform augmentation')
parser.add_argument('--weight_classes', action='store_true', help='use a weighted loss to account for class imbalance')
parser.add_argument('--num_epochs', type=int, default=5000, help='number of epochs of training')
parser.add_argument('--num_layers', type=int, default=2, help='number of hidden layers of the mlp')
parser.add_argument('--num_hidden', type=int, default=8, help='number of hidden units per layer')

args = parser.parse_args()
# parse environment parameters and instantiate the environment
features_idx = [0, 1, 2]
np.random.seed(args.seed)
#Ouput of Twitter Use Case Notebook
clustering = [['Carol', 'Frank', 'Olivia', 'Niaj'],
              ['Bob', 'Chuck', 'Craig', 'Dan', 'Erin', 'Grace', 'Heidi', 'Ivan', 'Mallory'],
              ['Mike']]

# settings to run and compare, together with their specific paramenters
setting_to_reqs = {
    'pgirl': (solve_sigma_PGIRL, {'seed': args.seed, 'girl': True, 'other_options': [False, True, False]}, []),
    #'ra_pgirl_diag': (solve_sigma_PGIRL, {"diag": True, 'seed': args.seed}, []),
    'ra_pgirl_cov_estimation': (solve_sigma_PGIRL, {"cov_estimation": True, 'seed': args.seed}, []),
}
settings = list(setting_to_reqs.keys())
if args.settings != '':
    settings = [x for x in args.settings.split(',')]
# compute optimal lqg policy in closed form
demonstrations = 'datasets/twitter/'
states_data = np.load(demonstrations + 'states.pkl', allow_pickle=True)
actions_data = np.load(demonstrations + 'actions.pkl', allow_pickle=True)
reward_data = np.load(demonstrations + 'rewards.pkl', allow_pickle=True)
start_time = str(time.time())
dir_to_read = args.dir_to_read
cluster_results = {}
for setting in settings:
    cluster_results[setting] = []

cluster_results['settings'] = settings
for i, cluster in enumerate(clustering):
    X_dataset = []
    y_dataset = []
    r_dataset = []
    for setting in settings:
        cluster_results[setting].append([[], []]) # weights , losses
    out_dir = 'logs/twitter/cluster_' + str(i) + '/' + start_time + "/"
    for agent in cluster:
        print("Collected data Agent: " + agent)
        X_dataset.append(states_data[agent][:len(states_data[agent]) // args.horizon * args.horizon])
        y_dataset.append(actions_data[agent][:len(states_data[agent]) // args.horizon * args.horizon])
        r_dataset.append(reward_data[agent][:len(states_data[agent]) // args.horizon * args.horizon])
        # dataset build
    states = np.concatenate(X_dataset)
    actions = np.concatenate(y_dataset)
    rewards = np.concatenate(r_dataset)
    for exp in range(args.num_experiments):
        np.random.seed(exp)
        pi, _ = bc_twitter(states, actions.flatten(), args, out_dir, out_dir)
        if not args.load_gradients:
            grads, _ = compute_gradient(pi, states, actions, rewards,  episode_length=args.horizon,
                                        discount_f=args.gamma, features_idx=features_idx, use_baseline=True,
                                        filter_gradients=True, discrete=True)
            if args.save_gradients:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                np.save(out_dir + '/estimated_gradients_' + str(exp + 1) + '.npy', grads)
        else:
            grads = np.load(dir_to_read + '/estimated_gradients_' + str(exp + 1) + '.npy')

        means = grads.mean(axis=0)
        for setting in settings:
            solver, params, result = setting_to_reqs[setting]
            weights, loss, _ = solver(grads, **params)
            result.append({
                "loss": loss,
                "weights": list(weights),
            })

            cluster_results[setting][i][0].append(weights)
            cluster_results[setting][i][1].append(loss)
save_dir = 'logs/twitter/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir + '/results_clustering.pkl', 'wb') as handle:
    pickle.dump(cluster_results, handle)
