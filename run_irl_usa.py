from algorithms.gpomdp import *
from algorithms.sigma_girl import *
import argparse
from bc.behavioural_cloning_como import behavioral_cloning_nn, behavioral_clonning_linear
import tensorflow as tf
from utils.utils import filter_grads
import pickle

dataset_number = 1817

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_number', type=int, default=1817, help='index of dataset')
parser.add_argument('--num_layers', type=int, default=0, help='number of layers of mlp network')
parser.add_argument('--num_hidden', type=int, default=8, help='number of units per layer')
parser.add_argument('--num_epochs', type=int, default=63, help='number of training epochs in bc')
parser.add_argument('--dir', type=str, default='/Users/Manuel/Desktop/irl_real_life/ResOpsUS/data/' + str(dataset_number) + '/', help='directory where to read the trajectories')
parser.add_argument('--starting_point', type=str, default='', help='path of the policy to start training from')
parser.add_argument('--validation', type=float, default=0.2, help='size of validation set')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate of the adam optimizer in the bc training')
parser.add_argument('--l2', type=float, default=0., help='l2 regularization in bc')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for bc')
parser.add_argument('--ep_len', type=int, default=-1, help='trajectory length')
parser.add_argument('--years_per_trajectory', type=int, default=1, help='how to split lifelong trajectory')
parser.add_argument('--years_step', type=float, default=1., help='can be used to make trajectories intersect')
parser.add_argument('--seed', type=int, default=-1, help='random seed')
parser.add_argument('--gamma', type=float, default=0.997, help='discount factor')
parser.add_argument('--init_logstd', type=float, default=2.5, help='initial policy variance')
parser.add_argument('--state_std', action='store_true', help='make the policy variance state dependent')
parser.add_argument('--norm_features', action='store_true', help='normalize state features')
parser.add_argument('--num_regimes', type=int, default=6, help='number of reward regimes')
parser.add_argument('--read_costs', action='store_true', help='read precomputed costs (Do not set if new features)')

args = parser.parse_args()

dataset_number = args.dataset_number
args.dir = '/Users/Manuel/Desktop/irl_real_life/ResOpsUS/data/' + str(args.dataset_number) + '/'


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

num_regimes = args.num_regimes
num_years = int(len(states_array) / 365)

X = states_array
Y = actions_array

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
logger_all['bc_losses']= logs['ll']

states = []
actions = []
features = []

base = 0
stepsize = 365
while base < num_years * 365:
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
        if args.num_layers == 0:
            step_layers, _, _, _, _ = pi.compute_gradients(st, action)
        else:
            step_layers, _, _, _ = pi.compute_gradients(st, action)
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

logger_all['irl_loss'] = {'Sigma-Girl-Full': None,
                                'Sigma-Girl-Diag': None,
                                'Sigma-Girl-Corrected': loss_cov,
                                'Girl': loss_pgirl}

logger_all['irl_weight'] = {'Sigma-Girl-Full': None,
                                 'Sigma-Girl-Diag': None,
                                 'Sigma-Girl-Corrected': values_cov,
                                 'Girl': values_pgirl}

with open('/Users/Manuel/Desktop/irl_real_life/logs/ResOpsUS/'+ str(dataset_number) +'/res_non_stationary.pickle', 'wb') as f:
    pickle.dump(logger_all, f, protocol=pickle.HIGHEST_PROTOCOL)

