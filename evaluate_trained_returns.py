import pickle
import numpy as np
from policies.rule_based import RuleBasedPolicy, extract_features

# To load the trained policies, will not use since we cannot share the environment implementation
def load_trained_pol(path):
    agent_weights = np.load(path)
    pi = RuleBasedPolicy()
    pi.set_params(agent_weights)
    pi.set_mean()

    def pol(ob):
        s = extract_features(ob, scaled=True)
        ac = pi.act(s)
        return ac
    return pol


def compute_return(features, weights, gamma=0.9999, horizon=400):
    rewards = np.dot(features, weights)
    discount_factor_timestep = np.power(gamma * np.ones(horizon),range(horizon))  # (T,)
    rewards = rewards.reshape([-1, horizon])
    discounted_return = discount_factor_timestep[np.newaxis, :] * rewards  # (N,T)
    disc_return = discounted_return.sum(axis=1).mean(axis=0)
    return disc_return


if __name__ == '__main__':

    agents = ['Bob', 'Alice', 'Carol', 'Chuck', 'Craig', 'Dan', 'Erin', 'Eve', 'Grace', 'Judy']
    base_dir = 'datasets/highway/'
    trained_dir = 'logs/highway/trained/'
    agent_to_evaluations = {}
    agent_to_model = {
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
    # take the values from the notebook where the fe clustering is made
    fe_clustering = [['Bob', 'Alice', 'Dan', 'Eve', 'Grace'],
                     ['Carol', 'Erin'],
                     ['Craig', 'Judy'],
                     ['Chuck']]
    irl_clustering = [['Eve', 'Grace', 'Alice'],
                      ['Carol', 'Erin', 'Bob', 'Dan', 'Chuck'],
                      ['Craig'],
                      ['Judy']]

    rewards = [[0.76, 0., 0.24],
               [0.09, 0., 0.91],
               [1., 0., 0.],
               [0.19, 0.81, 0.]]
    evals = []
    gamma = 0.9999
    horizon = 400
    cluster_agents = ['Cluster_1', 'Cluster_2',
                      'Cluster_3', 'Cluster_4', ]
    cluster_count = 0
    returns = []
    # We cannot give the environment to evaluate the policies so we report the values of trained policies evaluations
    # In the original code we would evaluate the policies in the simulator instead
    trained_returns = [-18.35, -50.19966302034527, -0.38, -18.05]
    for i, clustering in enumerate(irl_clustering):
        pol = load_trained_pol(trained_dir + cluster_agents[i] + '/best.npy')
        cluster_weights = rewards[i]
        trained_pol_return = trained_returns[i]
        data_dir = 'datasets/highway/irl/cluster_' + str(i)
        cluster_returns = []
        for j, agent in enumerate(clustering):
            data_path = base_dir + agent
            features_array = np.load(data_path + '/rewards.npy', allow_pickle=True)
            disc_return = compute_return(features_array, cluster_weights, gamma=gamma, horizon=horizon)
            cluster_returns.append(disc_return)
            print("Return cluster " + str(i) + " own agent " + str(j + 1) + " " + agent + ":" + str(disc_return))
        other_returns = []
        for agent in agents:
            agent_count = 0
            if agent not in clustering:
                data_path = base_dir + agent
                features_array = np.load(data_path + '/rewards.npy', allow_pickle=True)
                disc_return = compute_return(features_array, cluster_weights, gamma=gamma, horizon=horizon)
                other_returns.append(disc_return)
                print("Return cluster " + str(i) + " other agent " + str(agent_count + 1) + " " + agent + ":" +
                      str(disc_return))
                agent_count += 1
        returns.append((cluster_returns, other_returns, trained_pol_return))
        cluster_count += 1

    pickle.dump(returns, open('logs/highway/irl_returns', 'wb'))
