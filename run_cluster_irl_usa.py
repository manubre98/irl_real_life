import numpy as np
from bc.behavioural_cloning_como import behavioral_cloning_nn, behavioral_clonning_linear
from utils.utils import compute_gradient_usa, load_policy, estimate_distribution_params, filter_grads
from algorithms.clustering import em_clustering
import argparse
import pickle

# Directories where the agent policies, trajectories and gradients (if already calcualted) are stored
# To add agents populate this dictionary and store the gradients in '/gradients/estimated_gradients.npy'
# Or if u want to calculate the gradients directly store the policy as a tf checkpoint in a file called best
# and the trajectories in the subfolder 'trajectories/<subfolder>/K_trajectories.csv'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=0, help='number of hidden layers')
    parser.add_argument('--num_hidden', type=int, default=8, help='number of hidden units')
    parser.add_argument('--n_experiments', type=int, default=10, help='number of experiments')
    parser.add_argument('--gamma', type=float, default=0.997, help='discount factor')
    parser.add_argument('--verbose', action='store_true', help='print logs in console')
    parser.add_argument('--ep_len', type=int, default=365, help='episode length')
    parser.add_argument('--save_grad', action='store_true', help='save computed gradients')
    parser.add_argument('--mask', action='store_true', help='mask timesteps for baseline in gradient computation')
    parser.add_argument('--baseline', action='store_true', help='use baseline in gradient computation')
    parser.add_argument('--scale_features', type=int, default=1, help='rescale features in gradient computation')
    parser.add_argument('--filter_gradients', action='store_true', help='regularize jacobian matrix')
    parser.add_argument('--trainable_variance', action='store_true', help='fit the variance of the policy')
    parser.add_argument("--init_logstd", type=float, default=-1, help='initial policy variance')
    parser.add_argument('--save_path', type=str, default='/Users/Manuel/Desktop/irl_real_life/ResOpsUS/Results/', help='path to save the model')
    args = parser.parse_args()

    seed = 6
    np.random.seed(seed)

    n_experiments = args.n_experiments
    results = []
    n_agents = 1
    # where the demonstrations are
    demonstrations = 'ResOpsUS/data'
    dam_to_data = [55, 57, 60, 63, 87, 114, 116, 131, 132, 157, 165, 182, 198, 210, 292, 293, 295, 300, 317, 319, 320, 361, 362, 367, 368, 372, 374, 378, 382, 384, 385, 386, 398, 399, 413, 415, 416, 419, 421, 442, 445, 448, 449, 450, 469, 470, 471, 473, 477, 492, 493, 502, 506, 514, 517, 518, 519, 527, 531, 536, 546, 549, 554, 567, 572, 595, 601, 604, 654, 780, 784, 868, 872, 893, 936, 938, 939, 953, 956, 976, 931, 1095, 1242, 192, 373, 545, 987, 1109, 438, 456, 657, 898, 310, 542, 41, 42, 80, 89, 93, 204, 297, 305, 364, 405, 423, 541, 597, 616, 913, 920, 929, 1044, 1053, 1112, 1269, 1283, 1291, 1292, 1297, 1752, 1756, 1758, 1768, 1787, 1817, 1823, 1834, 1835, 1851, 1872, 7306, 7308, 7313, 7318, 1754, 1774, 1801, 1879, 1896, 133, 299, 316, 993, 355, 798, 837, 962, 1140, 1280, 1294, 1296, 85, 989, 1761, 1762, 1800, 1186, 1194, 1220, 1236, 1247, 1277, 1302, 1776, 1782, 56, 88, 90, 92, 97, 99, 100, 107, 148, 307, 338, 390, 391, 393, 511, 613, 740, 753, 777, 836, 854, 861, 870, 884, 895, 924, 948, 952, 955, 957, 958, 961, 963, 964, 965, 967, 968, 972, 974, 975, 979, 980, 981, 982, 983, 991, 998, 1000, 1001, 1003, 1006, 1007, 1014, 1016, 1017, 1019, 1020, 1021, 1023, 1026, 1027, 1028, 1032, 1033, 1036, 1042, 1048, 1050, 1060, 1067, 1077, 1084, 1086, 1092, 1093, 1101, 1120, 1121, 1122, 1123, 1124, 1125, 1134, 1135, 1144, 1237, 1586, 1587, 1592, 1600, 1606, 1617, 1619, 1620, 1631, 1636, 1645, 1650, 1654, 1655, 1659, 1683, 1691, 1699, 1703, 1706, 1707, 1709, 1712, 1713, 1714, 1716, 1718, 1723, 1726, 1733, 1735, 1739, 1740, 1741, 1742, 1744, 1755, 1763, 1765, 1767, 1770, 1775, 1777, 1781, 1818, 1828, 1833, 1841, 1843, 1846, 1848, 1855, 1862, 1863, 1864, 1869, 7311, 7317, 163, 169, 600, 907, 911, 969, 1070, 1128, 1183, 1202, 1585, 1615, 2193]

    num_objectives = 3
    num_clusters = 10
    #states_data = np.load(demonstrations + 'real_states4.pkl', allow_pickle=True)
    #actions_data = np.load(demonstrations + 'actions2.pkl', allow_pickle=True)
    #reward_data = np.load(demonstrations + 'rewards4.pkl', allow_pickle=True)
    features_idx = [0, 1, 2]
    GAMMA = args.gamma
    for exp in range(n_experiments):
        P_true = np.array([[1, 1, 0, 0, 0] * n_agents, [0, 0, 1, 1, 1] * n_agents], dtype=np.float)
        print("Experiment %s" % (exp+1))
        estimated_gradients_all = []
        for dam_number in dam_to_data:
            base_dir = '/Users/Manuel/Desktop/irl_real_life/ResOpsUS/data/' + str(dam_number) + "/"
            X_dataset = np.load(base_dir + 'states.npy', allow_pickle=True)
            y_dataset = np.load(base_dir + 'actions.npy', allow_pickle=True)
            r_dataset = np.load(base_dir+ 'reward_features.npy', allow_pickle=True)

            X_dim = len(X_dataset[0])
            y_dim = y_dataset.shape[-1]

            if args.num_layers == 0:
                pi, logs, model_path = behavioral_clonning_linear(X_dataset, y_dataset, noise=np.exp(args.init_logstd))
                print(pi.weights, pi.bias)
            else:
                pi, logs, model_path = behavioral_cloning_nn(num_epochs=args.num_epochs, num_hidden=args.num_hidden,
                                                             num_layers=args.num_layers, X=X_dataset, Y=y_dataset,
                                                             validation=args.validation, lr=args.lr, l2=args.l2,
                                                             batch_size=args.batch_size,
                                                             init_logstd=args.init_logstd,
                                                             state_dependent_variance=args.state_std,
                                                             starting_point=args.starting_point, )

            # Create Policy
            #model = 'bc/models/' + agent_name + '/2000_22/best'
            #linear = 'gpomdp' in model
            #print('load policy..')
            #policy_train = load_policy(X_dim=X_dim, model=model, continuous=False, num_actions=y_dim,
            #                           n_bases=X_dim,
            #                           trainable_variance=args.trainable_variance, init_logstd=args.init_logstd,
            #                           linear=linear, num_hidden=args.num_hidden, num_layers=args.num_layers)
            #print('Loading dataset... done')
            # compute gradient estimation
            estimated_gradients = compute_gradient_usa(pi, X_dataset, y_dataset, r_dataset,
                                                      args.ep_len, GAMMA, features_idx,
                                                      verbose=args.verbose,
                                                      use_baseline=args.baseline,
                                                      use_mask=args.mask,
                                                      scale_features=args.scale_features,
                                                      filter_gradients=args.filter_gradients,
                                                      normalize_f=False)
            estimated_gradients_all.append(estimated_gradients)
        # ==================================================================================================================

            if args.save_grad:
                print("Saving gradients in ", args.save_path)
                np.save(args.save_path + '/estimated_gradients.npy', estimated_gradients)
        mus = []
        sigmas = []
        ids = []

        for i, dam_number in enumerate(dam_to_data):
            num_episodes, num_parameters, num_objectives = estimated_gradients_all[i].shape[:]
            mu, sigma = estimate_distribution_params(estimated_gradients=estimated_gradients_all[i],
                                                    diag=False, identity=False, other_options=[False, False],
                                                    cov_estimation=True)
            id_matrix = np.identity(num_parameters)
            mus.append(mu)
            sigmas.append(sigma)
            ids.append(id_matrix)

        P, Omega, loss, prob = em_clustering(mus, sigmas, ids, num_clusters=num_clusters,
                                       num_objectives=num_objectives,
                                       optimization_iterations=1)
        print(P)
        results.append((P, Omega, loss, prob))
    with open(args.save_path + f'/results_new_total_one_{num_clusters}.pkl', 'wb') as handle:
        pickle.dump(results, handle)