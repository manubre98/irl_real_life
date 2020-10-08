from bc.behavioural_cloning_highway import load_policy
import pickle
import numpy as np
agents = ['Bob', 'Alice', 'Carol', 'Chuck', 'Craig', 'Dan', 'Erin', 'Eve', 'Grace', 'Judy']
base_dir = 'datasets/highway/'
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
agent_to_evaluations = {}


fe_clustering = [['Bob', 'Alice', 'Dan', 'Eve', 'Grace'],
                     ['Carol', 'Erin'],
                     ['Craig', 'Judy'],
                     ['Chuck']]
irl_clustering = [['Eve', 'Grace', 'Alice'],
                  ['Carol', 'Erin', 'Bob', 'Dan', 'Chuck'],
                  ['Craig'],
                  ['Judy']]
evals = []
cluster_count = 0
for i, clustering in enumerate(irl_clustering):
    if len(clustering) == 1:
        continue
    evals.append([])
    cluster_evals = []
    for agent in clustering:
        cluster_actions = []
        cluster_states = []
        for agent2 in clustering:
            if agent != agent2:
                data_path = base_dir + agent2
                states_array = np.load(data_path + '/states.npy', allow_pickle=True)
                actions_array = np.load(data_path + '/actions.npy', allow_pickle=True)
                cluster_actions.append(actions_array)
                cluster_states.append(states_array)
        cluster_actions = np.concatenate(cluster_actions, axis=0)
        cluster_states = np.concatenate(cluster_states, axis=0)
        print("Doing agent:" + agent)
        model_path = agent_to_model[agent][0] + '/best'
        data_path = base_dir + agent
        state_dim = cluster_states.shape[-1]

        pi, evaluations, distributions = load_policy(model_path , state_dim, 3, 8, 1,
                    init_logstd=0, discrete=True, beta=1, X=cluster_states, Y=cluster_actions)
        pi.sess.close()
        print("Dist:" + str(distributions))
        agent_to_evaluations[agent] = [evaluations, distributions]
        evals[cluster_count].append(evaluations)
    cluster_count += 1

pickle.dump(agent_to_evaluations, open('logs/highway/irl_evaluations', 'wb'))
pickle.dump(evals, open('logs/highway/irl_evaluations_list', 'wb'))
