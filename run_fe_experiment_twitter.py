import numpy as np
from utils.utils import feature_expectations
import argparse
import pickle
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--ep_len', type=int, default=20, help='episode length')
    parser.add_argument('--save_path', type=str, default='logs/twitter/feature_exp', help='path to save the model')
    args = parser.parse_args()
    results = []
    # where the demonstrations are
    demonstrations = 'datasets/twitter/'
    agent_to_data = ['Bob', 'Carol', 'Chuck', 'Craig', 'Dan', 'Erin', 'Grace', 'Frank', 'Heidi', 'Ivan', 'Mallory',
                     'Mike', 'Olivia', 'Niaj']
    num_objectives = 3
    states_data = np.load(demonstrations + 'states.pkl', allow_pickle=True)
    actions_data = np.load(demonstrations + 'actions.pkl', allow_pickle=True)
    reward_data = np.load(demonstrations + 'rewards.pkl', allow_pickle=True)
    GAMMA = args.gamma
    feat_expecations_dict = {}
    feat_expecations_list = []
    for i, agent_name in enumerate(agent_to_data):
        print('Doing Agent:', agent_name)

        X_dataset = states_data[agent_name]
        y_dataset = actions_data[agent_name]
        r_dataset = reward_data[agent_name]
        print(len(X_dataset))
        # Create Policy
        r_dataset = np.array(r_dataset[:len(r_dataset) // args.ep_len * args.ep_len]).reshape([-1, args.ep_len,
                                                                                               num_objectives])
        feat_expectations = feature_expectations(r_dataset, GAMMA).mean(axis=0)
        feat_expecations_dict[agent_name] = feat_expectations
        feat_expecations_list.append(feat_expectations)

    results = {
        'agent_list': agent_to_data,
        'feature_list': feat_expecations_list,
        'feature_dict': feat_expecations_dict,
    }

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(args.save_path + '/results.pkl', 'wb') as handle:
        pickle.dump(results, handle)
