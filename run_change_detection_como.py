from algorithms.gpomdp import *
from algorithms.sigma_girl import *
import argparse
from bc.behavioural_cloning_como import behavioral_cloning_nn, behavioral_clonning_linear
import tensorflow as tf
from utils.utils import filter_grads
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', type=int, default=0, help='number of layers of mlp network')
parser.add_argument('--num_hidden', type=int, default=8, help='number of units per layer')
parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs in bc')
parser.add_argument('--dir', type=str, default='datasets/como/', help='directory where to read the trajectories')
parser.add_argument('--starting_point', type=str, default='', help='path of the policy to start training from')
parser.add_argument('--validation', type=float, default=0., help='size of validation set')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate of the adam optimizer in the bc training')
parser.add_argument('--l2', type=float, default=0., help='l2 regularization in bc')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for bc')
parser.add_argument('--ep_len', type=int, default=-1, help='trajectory length')
parser.add_argument('--years_per_trajectory', type=int, default=1, help='how to split lifelong trajectory')
parser.add_argument('--years_step', type=float, default=1., help='can be used to make trajectories intersect')
parser.add_argument('--seed', type=int, default=-1, help='random seed')
parser.add_argument('--gamma', type=float, default=0.9999, help='discount factor')
parser.add_argument('--init_logstd', type=float, default=2.5, help='initial policy variance')
parser.add_argument('--state_std', action='store_true', help='make the policy variance state dependent')
parser.add_argument('--norm_features', action='store_true', help='normalize state features')
parser.add_argument('--num_regimes', type=int, default=10, help='number of reward regimes')
parser.add_argument('--read_costs', action='store_true', help='read precomputed costs (Do not set if new features)')

args = parser.parse_args()

logger_all = {'num_layers': args.num_layers,
              'num_hidden': args.num_hidden,
              'state_std': args.state_std,
              'num_regimes': args.num_regimes}

states_array = np.load(args.dir + 'states.npy', allow_pickle=True)
actions_array = np.load(args.dir + 'actions.npy', allow_pickle=True)
features_array = np.load(args.dir + 'reward_features.npy', allow_pickle=True)

state_dim = states_array.shape[-1]
action_dim = actions_array.shape[-1]

seed = None
if args.seed != -1:
    seed = args.seed
np.random.seed(seed)
s = np.random.randint(0, 100000)
tf.set_random_seed(s)

if args.ep_len != -1 and 365 % args.ep_len == 0:
    ep_len = args.ep_len
elif args.years_per_trajectory > 0:
    ep_len = int(365 * args.years_per_trajectory)
else:
    ep_len = 365


# define the loss function
def loss(u, v):
    X = states_array[u * 365:v * 365]
    Y = actions_array[u * 365:v * 365]
    if args.num_layers == 0:
        pi, logs, model_path = behavioral_clonning_linear(X, Y, noise=np.exp(args.init_logstd))
    else:
        pi, logs, model_path = behavioral_cloning_nn(num_epochs=args.num_epochs, num_hidden=args.num_hidden,
                                                     num_layers=args.num_layers, X=X, Y=Y, validation=args.validation,
                                                     lr=args.lr, l2=args.l2, batch_size=args.batch_size,
                                                     init_logstd=args.init_logstd,
                                                     state_dependent_variance=args.state_std,
                                                     starting_point=args.starting_point)
    states = []
    actions = []
    features = []
    base = 365 * u
    stepsize = 365
    while base < v * 365:
        st = states_array[base:base + ep_len]
        act = actions_array[base:base + ep_len]
        ft = features_array[base:base + ep_len]
        states.append(st)
        actions.append(act)
        features.append(ft)
        base += stepsize
    states = np.array(states)
    actions = np.array(actions)
    features = np.array(features)

    grads = []
    print(states.shape)
    for i in range(states.shape[0]):
        grads.append([])
        for j in range(states.shape[1]):
            st = states[i, j]
            action = actions[i, j]
            step_layers, _, _, _, _ = pi.compute_gradients(st, action)
            step_gradients = []
            for layer in step_layers:
                step_gradients.append(layer.ravel())
            step_gradients = np.concatenate(step_gradients)

            if np.isnan(step_gradients).any():
                print("NAN Grad")
            if (np.abs(step_gradients) > 10000).any():
                print("Big Grad")
            grads[i].append(step_gradients.tolist())
    gradients = np.array(grads)
    G = GPOMDP(gradients, gamma=args.gamma, rewards=features)
    gradients = G.eval_gpomdp(normalize_features=args.norm_features)
    gradients = filter_grads(gradients, verbose=True)

    values_cov, loss_cov, _ = solve_sigma_PGIRL(gradients, cov_estimation=True)
    # compute loss
    return loss_cov


num_regimes = args.num_regimes
num_years = 65
cost = np.ones((args.num_regimes, num_years, num_years)) * np.inf

if args.read_costs:
    try:
        cost = np.load('logs/como/cost_again.npy')
    except:
        # Initialize the costs
        for u in range(num_years - 1):
            for v in range(u, num_years):
                cost[0, u, v] = loss(u, v + 1) * (v - u + 1)
        np.save('logs/como/cost_again.npy', cost)
else:
    for u in range(num_years - 1):
        for v in range(u, num_years):
            cost[0, u, v] = loss(u, v + 1) * (v - u + 1)
    np.save('logs/como/cost_again.npy', cost)

for num_regimes in np.arange(1, 11, dtype=int):
    for k in range(2, num_regimes):
        k_idx = k - 1
        for u in range(num_years - k + 1):
            for v in range(u + k - 1, num_years):
                candidates = [cost[k_idx - 1, u, t] + cost[0, t + 1, v] for t in range(u + k - 2, v)]
                cost[k_idx, u, v] = np.min(candidates)

    # Find idxs
    L = np.zeros(num_regimes, dtype=int)
    L[num_regimes - 1] = num_years - 1
    for k_idx in range(num_regimes - 1, 0, -1):
        s = L[k_idx]
        candidates = [cost[k_idx - 1, 0, t] + cost[0, t + 1, s] for t in range(k_idx - 1, s)]
        t_star = np.argmin(candidates) + k_idx - 1
        L[k_idx - 1] = t_star

    L = np.concatenate(([-1], L))
    L_start = L[:-1] + 1
    L_stop = L[1:]

    logger_all['intervals'] = (L_start, L_stop)
    logger_all['bc_losses'] = np.zeros(num_regimes)
    logger_all['irl_weights'] = [{} for _ in range(num_regimes)]
    logger_all['irl_losses'] = [{} for _ in range(num_regimes)]

    total_loss = 0.
    for u, v in zip(L_start, L_stop):
        print(u, v, cost[0, u, v])
        total_loss += cost[0, u, v]

    print('Loss Change Point Detection:', total_loss)
    print('Loss No Change Point Detection:', cost[0, 0, -1])

    logger_all['irl_total_loss'] = total_loss
    logger_all['irl_total_loss_no_detection'] = cost[0, 0, -1]
    logger_all['bc_total_loss'] = 0.
    for ii, (u, v) in enumerate(zip(L_start, L_stop + 1)):
        print('INTERVAL: [%s, %s]' % (u, v))
        X = states_array[u * 365:v * 365]
        Y = actions_array[u * 365:v * 365]

        if args.num_layers == 0:
            pi, logs, model_path = behavioral_clonning_linear(X, Y, noise=np.exp(args.init_logstd))
            print(pi.weights, pi.bias)
        else:
            pi, logs, model_path = behavioral_cloning_nn(num_epochs=args.num_epochs, num_hidden=args.num_hidden,
                                                         num_layers=args.num_layers, X=X, Y=Y,
                                                         validation=args.validation, lr=args.lr, l2=args.l2,
                                                         batch_size=args.batch_size,
                                                         init_logstd=args.init_logstd,
                                                         state_dependent_variance=args.state_std,
                                                         starting_point=args.starting_point,)
        logger_all['bc_losses'][ii] = logs['ll']
        logger_all['bc_total_loss'] += logs['ll']

        states = []
        actions = []
        features = []

        base = 365 * u
        stepsize = 365
        while base < v * 365:
            st = states_array[base:base + ep_len]
            act = actions_array[base:base + ep_len]
            ft = features_array[base:base + ep_len]
            states.append(st)
            actions.append(act)
            features.append(ft)
            base += stepsize

        states = np.array(states)
        actions = np.array(actions)
        features = np.array(features)

        grads = []
        for i in range(states.shape[0]):
            grads.append([])
            for j in range(states.shape[1]):
                st = states[i, j]
                action = actions[i, j]
                step_layers, _, _, _, _ = pi.compute_gradients(st, action)
                step_gradients = []
                for layer in step_layers:
                    step_gradients.append(layer.ravel())
                step_gradients = np.concatenate(step_gradients)

                if np.isnan(step_gradients).any():
                    print("NAN Grad")
                if (np.abs(step_gradients) > 10000).any():
                    print("Big Grad")
                grads[i].append(step_gradients.tolist())
        gradients = np.array(grads)
        G = GPOMDP(gradients, gamma=args.gamma, rewards=features)
        gradients = G.eval_gpomdp(normalize_features=args.norm_features)
        gradients = filter_grads(gradients, verbose=True)

        values_cov, loss_cov, _ = solve_sigma_PGIRL(gradients, cov_estimation=True)
        values_pgirl, loss_pgirl, _ = solve_sigma_PGIRL(gradients, girl=True)
        print("Sigma Girl Corrected: \t Weights:" + str(values_cov) + '\t Loss:' + str(loss_cov))
        print("Girl: \t Weights:" + str(values_pgirl) + '\t Loss:' + str(loss_pgirl))

        logger_all['irl_losses'][ii] = {'Sigma-Girl-Full': None,
                                        'Sigma-Girl-Diag': None,
                                        'Sigma-Girl-Corrected': loss_cov,
                                        'Girl': loss_pgirl}

        logger_all['irl_weights'][ii] = {'Sigma-Girl-Full': None,
                                         'Sigma-Girl-Diag': None,
                                         'Sigma-Girl-Corrected': values_cov,
                                         'Girl': values_pgirl}

    with open('logs/como/res_non_stationary_%s.pickle' % num_regimes, 'wb') as f:
        pickle.dump(logger_all, f, protocol=pickle.HIGHEST_PROTOCOL)

