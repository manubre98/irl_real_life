from bc.behavioural_cloning_highway import load_policy
import pickle
import numpy as np


def compute_feature_expectations(features, gamma=0.9999, horizon=400):
    features = features.reshape([-1, horizon, features.shape[-1]])
    discount_factor_timestep = np.power(gamma * np.ones(horizon), range(horizon))  # (T,)
    feature_expectations = discount_factor_timestep[np.newaxis, :, np.newaxis] * features  # (N,T,Q)
    feature_expectations = feature_expectations.sum(axis=1).mean(axis=0)
    return feature_expectations.flatten()


agents = ['Bob', 'Alice', 'Carol', 'Chuck', 'Craig', 'Dan', 'Erin', 'Eve', 'Grace', 'Judy']
base_dir = 'datasets/highway/'
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
feature_expectations = []
gamma = 0.9999
horizon = 400
for agent in agents:
    print("Doing agent:" + agent)
    model_path = agent_to_model[agent][0] + '/best'
    data_path = base_dir + agent
    states_array = np.load(data_path + '/states.npy', allow_pickle=True)
    actions_array = np.load(data_path + '/actions.npy', allow_pickle=True)
    rewards_array = np.load(data_path + '/rewards.npy', allow_pickle=True)
    state_dim = states_array.shape[-1]
    pi, evaluations, distributions = load_policy(model_path , state_dim, 3, 8, 1, init_logstd=0, discrete=True, beta=1,
                                                 X=states_array, Y=actions_array)
    pi.sess.close()
    print("Dist:" + str(distributions))
    fe = compute_feature_expectations(rewards_array, gamma=gamma, horizon=horizon)
    feature_expectations.append(fe)
    agent_to_evaluations[agent] = [evaluations, distributions]

pickle.dump(agent_to_evaluations, open('logs/highway/evaluations', 'wb'))
pickle.dump(feature_expectations, open('logs/highway/feature_expectations', 'wb'))
