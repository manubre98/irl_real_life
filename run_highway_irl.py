from algorithms.gpomdp import *
from algorithms.sigma_girl import *
import argparse
from bc.behavioural_cloning_highway import behavioral_cloning_nn, behavioral_clonning_linear, load_policy
import tensorflow as tf
from utils.utils import filter_grads

parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', type=int, default=1, help='number of layers of mlp network')
parser.add_argument('--num_hidden', type=int, default=8, help='number of units per layer')
parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs in bc')
parser.add_argument('--dir', type=str, default='datasets/highway/', help='directory where the datasets are stored')
parser.add_argument('--agent', type=str, default='Dan', help='name of the agent to perform irl on')
parser.add_argument('--model_path', type=str, default='', help='path of the pretrained policy')
parser.add_argument('--starting_point', type=str, default='', help='path of the policy to start training from')
parser.add_argument('--validation', type=float, default=0., help='portion of validation set in bc')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate of the adam optimizer in the bc training')
parser.add_argument('--l2', type=float, default=0., help='l2 regularization in bc')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for bc')
parser.add_argument('--verbose', action='store_true', help='log gradient messages')
parser.add_argument('--seed', type=int, default=-1, help='random seed')
parser.add_argument('--gamma', type=float, default=0.9999, help='discount factor')
parser.add_argument('--beta', type=float, default=1.0, help='temperature of the Boltzman policy')
parser.add_argument('--read_grad', action='store_true', help='read precomputed gradients')
parser.add_argument('--save_grad', action='store_true', help='save computed gradients')
parser.add_argument('--grad_path', type=str, default='datasets/second/gradients/grads.npy',
                    help='path to read gradients from')
parser.add_argument('--features_idx', default='', type=str, help='indexes of the features to perform irl on')
parser.add_argument('--norm_features', action='store_true', help='normalize the reward features')

args = parser.parse_args()
feature_labels = np.array(['r_freeright', 'r_lanechange', 'r_distancefront'])
dir = args.dir + args.agent + '/'
states_array = np.load(dir + 'states.npy', allow_pickle=True)
actions_array = np.load(dir + 'actions.npy', allow_pickle=True)
features_array = np.load(dir + 'rewards.npy', allow_pickle=True)
if args.features_idx == '':
    features_idx = [x for x in range(features_array.shape[-1])]
else:
    features_idx = [int(x) for x in args.features_idx.split(',')]
num_features = features_array.shape[-1]
state_dim = states_array.shape[-1]
action_dim = np.max(np.unique(actions_array)) + 1

seed = None
if args.seed != -1:
    seed = args.seed
np.random.seed(seed)
s = np.random.randint(0, 100000)
tf.set_random_seed(s)
ep_len = 400
if not args.read_grad:
    # compute gradients of features
    if args.model_path != '':
        #  load the pre trained policy
        pi, _, _ = load_policy(args.model_path, state_dim, action_dim, args.num_hidden, args.num_layers, discrete=True,
                               beta=args.beta, X=states_array,
                               Y=actions_array)
        model_path = '/'.join(args.model_path.split('/')[:-1])
    else:
        #  perform behavioral cloning
        if args.num_layers == 0:
            pi, model_path = behavioral_clonning_linear(states_array, actions_array, name='highway/' + args.agent + '/')
        else:
            pi, logs, model_path = behavioral_cloning_nn(num_epochs=args.num_epochs, num_hidden=args.num_hidden,
                                                         num_layers=args.num_layers, X=states_array, Y=actions_array,
                                                         validation=args.validation, lr=args.lr, l2=args.l2,
                                                         batch_size=args.batch_size,
                                                         starting_point=args.starting_point,
                                                         name='highway/' + args.agent + '/')
    # perform irl
    states = []
    actions = []
    features = []
    base = 0
    # split the trajectories in separate epissodes
    while base + ep_len <= states_array.shape[0]:
        st = states_array[base:base+ep_len]
        act = actions_array[base:base + ep_len]
        ft = features_array[base:base + ep_len]
        states.append(st)
        actions.append(act)
        features.append(ft)
        base += ep_len
    states = np.array(states)
    actions = np.array(actions)
    features = np.array(features)

    grads = []
    print(states.shape)
    count = 0
    for i in range(states.shape[0]):
        grads.append([])
        for j in range(states.shape[1]):
            st = states[i, j]
            action = actions[i, j]
            st = st.reshape(1, -1)
            action = [action]
            step_layers, prob, _, probs = pi.compute_gradients(st, [action])
            if prob != np.max(probs):
                count+= 1
            step_gradients = []
            for layer in step_layers:
                step_gradients.append(layer.ravel())
            step_gradients = np.concatenate(step_gradients)

            if np.isnan(step_gradients).any():
                print("NAN Grad")
            if (np.abs(step_gradients) > 10000).any():
                print("Big Grad")
            grads[i].append(step_gradients.tolist())
    print("Action si not the best in %d times out of %d in total:" %(count, states_array.shape[0]))
    gradients = np.array(grads)
    G = GPOMDP(gradients, gamma=args.gamma, rewards=features)
    gradients = G.eval_gpomdp(normalize_features=args.norm_features)
    gradients = filter_grads(gradients, verbose=True)

    if args.save_grad:
        np.save(model_path + '/gradients.npy', gradients)
else:
    gradients = np.load(args.grad_path)
    model_path = '/'.join(args.grad_path.split('/')[:-1])

gradients = gradients[:, :, features_idx]

feature_labels = feature_labels[features_idx]
print("Features:")
print(feature_labels)
#  perform irl

values_cov, loss_cov, _ = solve_sigma_PGIRL(gradients, cov_estimation=True,)
values_pgirl, loss_pgirl = solve_PGIRL(gradients)

print("Sigma Girl Corrected: \t Weights:" + str(values_cov) + '\t Loss:' + str(loss_cov))
print("Girl: \t Weights:" + str(values_pgirl) + '\t Loss:' + str(loss_pgirl))
np.save(model_path + '/girl.npy', values_pgirl)
np.save(model_path + '/sigma_cov.npy', values_cov)
