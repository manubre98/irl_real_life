import os
import sys
import tensorflow as tf
import numpy as np
from tqdm import trange
import random
import time
from gym.spaces import Box, Discrete
from policies.linear_gaussian_policy import LinearBoltzmanPolicy
path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path
from baselines.common.policies_bc import build_policy
from baselines.common.models import mlp
import baselines.common.tf_util as U
from sklearn.linear_model import  LogisticRegression


def behavioral_cloning_nn(num_epochs, num_layers, num_hidden, X, Y, validation=0.2, lr=1e-4, l2=0., batch_size=128,
                          starting_point='', name='', beta=1.0):
    input_dim = X.shape[-1]
    observation_space = Box(low=-np.inf, high=np.inf, shape=(input_dim,))
    action_space = Discrete(n=np.max(np.unique(Y))+1)
    tf.reset_default_graph()
    config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8,
        device_count={'CPU': 8}
    )

    config.gpu_options.allow_growth = True
    sess = U.make_session(make_default=True, config=config)
    network = mlp(num_hidden=num_hidden, num_layers=num_layers)
    policy_train = build_policy(observation_space, action_space, network, l2=l2, lr=lr, beta=beta)()
    U.initialize()
    if starting_point != '':
        policy_train.load(starting_point)
    model_name = str(num_epochs) + '_' + str(num_layers)+ '_' + str(num_hidden)
    tf_path = 'logs/tensorboards/' + name + '/' + model_name + '_' + str(time.time()) + '/'
    writer = tf.summary.FileWriter(tf_path)

    states = np.array(X)
    actions = np.array(Y)
    nops_state = states[actions == 0]
    right_state = states[actions == 2]
    left_state = states[actions == 1]
    not_nops_state = np.concatenate([right_state, left_state])
    nops_action = actions[actions == 0]
    right_action = actions[actions == 2]
    left_action = actions[actions == 1]
    not_nops_action = np.concatenate([right_action, left_action])
    dataset_not_nops = list(zip(np.array(not_nops_state), np.array(not_nops_action)))
    # check accuracy on lane changes only since they are a tiny percentage of the action performed
    X_val, y_val = zip(*dataset_not_nops)

    print("Original Dataset Size:", states.shape[0])
    classes = np.unique(Y)
    class_counts = np.array([np.sum(Y == cl) for cl in classes])
    max_count = max(class_counts)
    ratios = class_counts / max_count
    print("Class Distribution:", class_counts / states.shape[0])
    print("Class ratios:", ratios)
    states_to_add = []
    actions_to_add = []
    for j, ratio in enumerate(ratios):
        if ratio != 1:
            for i in range(int(1 / ratio)):
                states_to_add += states[actions == classes[j]].tolist()
                actions_to_add += actions[actions == classes[j]].tolist()
            remaining = int((1 / ratio - int(1 / ratio)) * class_counts[j])
            all_indexes = np.array([x for x in range(class_counts[j])])
            random.shuffle(all_indexes)
            shuffled_indexes = all_indexes[0:remaining]
            states_to_add += states[actions == classes[j]][shuffled_indexes].tolist()
            actions_to_add += actions[actions == classes[j]][shuffled_indexes].tolist()
    states_to_add = np.array(states_to_add)
    actions_to_add = np.array(actions_to_add)
    states = np.concatenate([states, states_to_add], axis=0)
    actions = np.concatenate([actions, actions_to_add], axis=0)
    print("Oversampled Dataset Size", states.shape[0])
    logger = {
        'batch_size': batch_size,
        'num_epochs': num_epochs
    }

    dataset = list(zip(states, actions))
    random.shuffle(dataset)
    dataset_training = dataset[:]

    # pre-processing statistics
    num_batches = len(dataset_training) // batch_size
    num_batches += (0 if len(dataset_training) % batch_size == 0 else 1)
    print('# batches: ', num_batches)
    print('# training samples: ', len(dataset_training))
    logger['num_batches'] = num_batches
    logger['training_samples'] = len(dataset_training)
    counter = 0
    for epoch in trange(num_epochs):
        #train batches built
        random.shuffle(dataset_training)
        batches = []
        for i in range(num_batches):
            base = batch_size * i
            batches.append(dataset_training[base: base + batch_size])
        if validation > 0. or True:
            accuracy, _, loss, _ = policy_train.evaluate(X_val[:], y_val[:], False)
            summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy validation", simple_value=accuracy), ])
            writer.add_summary(summary, epoch)
        else:
            policy_train.save(tf_path + '/best')
        for batch in batches:
            batch_X, batch_y = zip(*batch)
            target = batch_y
            output = policy_train.fit(batch_X, target)
            summaries = [tf.Summary.Value(tag="loss", simple_value=output[0]),
                         tf.Summary.Value(tag="r2", simple_value=output[1])]
            summaries += [tf.Summary.Value(tag="entropy", simple_value=output[2]),
                              tf.Summary.Value(tag="stochastic_accuracy", simple_value=output[3])]
            summary = tf.Summary(value=summaries)
            writer.add_summary(summary, counter)
            counter += 1
        # validation
    if validation > 0.:
        accuracy, _, loss, _ = policy_train.evaluate(X_val[:], y_val[:], False)
        summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy no nops", simple_value=accuracy), ])
        writer.add_summary(summary, epoch)
    if validation > 0:
        policy_train.save(tf_path + '/best')
    else:
        policy_train.save(tf_path + '/best')
    accuracy, _, loss, _ = policy_train.evaluate(nops_state, nops_action, False)
    print("Nops Accuracy:", accuracy)
    if right_state.shape[0] > 0:
        accuracy, _, loss, _ = policy_train.evaluate(right_state, right_action, False)
        print("Right Accuracy:", accuracy)

    if left_state.shape[0] > 0:
        accuracy, _, loss, _ = policy_train.evaluate(left_state, left_action, False)
        print("Left Accuracy:", accuracy)
    return policy_train, logger, tf_path


def behavioral_clonning_linear(X, Y, name=''):
    regr = LogisticRegression(fit_intercept=False)
    states = np.array(X)
    actions = np.array(Y)
    nops_state = states[actions == 0]
    right_state = states[actions == 2]
    left_state = states[actions == 1]
    nops_action = actions[actions == 0]
    right_action = actions[actions == 2]
    left_action = actions[actions == 1]

    print("Original Dataset Size:", states.shape[0])
    classes = np.unique(Y)
    class_counts = np.array([np.sum(Y == cl) for cl in classes])
    max_count = max(class_counts)
    ratios = class_counts / max_count
    print("Class Distribution:", class_counts / states.shape[0])
    print("Class ratios:", ratios)
    states_to_add = []
    actions_to_add = []
    for j, ratio in enumerate(ratios):
        if ratio != 1:
            for i in range(int(1 / ratio)):
                states_to_add += states[actions == classes[j]].tolist()
                actions_to_add += actions[actions == classes[j]].tolist()
            remaining = int((1 / ratio - int(1 / ratio)) * class_counts[j])
            all_indexes = np.array([x for x in range(class_counts[j])])
            random.shuffle(all_indexes)
            shuffled_indexes = all_indexes[0:remaining]
            states_to_add += states[actions == classes[j]][shuffled_indexes].tolist()
            actions_to_add += actions[actions == classes[j]][shuffled_indexes].tolist()
    states_to_add = np.array(states_to_add)
    actions_to_add = np.array(actions_to_add)
    states = np.concatenate([states, states_to_add], axis=0)
    actions = np.concatenate([actions, actions_to_add], axis=0)
    print("Oversampled Dataset Size", states.shape[0])
    train_data = states
    train_label = actions
    regr.fit(train_data, train_label)
    pi = LinearBoltzmanPolicy(weights=regr.coef_.T)
    model_name = 'linear'
    model_path = 'logs/tensorboards/' + name + '/' + model_name + '_' + str(time.time()) + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    np.save(model_path + 'best.npy', regr.coef_)
    print("R2 Score:" + str(regr.score(X, Y)))
    accuracy = regr.score(nops_state, nops_action)
    print("Nops Accuracy:", accuracy)
    accuracy = regr.score(right_state, right_action)
    print("Right Accuracy:", accuracy)
    accuracy = regr.score(left_state, left_action)
    print("Left Accuracy:", accuracy)
    return pi, model_path


def load_policy(model_path, input_dim, output_dim, num_hidden, num_layers, init_logstd=1., discrete=False,
                beta=1.0, use_bias=True, X=None, Y=None):
    observation_space = Box(low=-np.inf, high=np.inf, shape=(input_dim,))
    if discrete:
        action_space = Discrete(n=output_dim)
    else:
        action_space = Box(low=-np.inf, high=np.inf, shape=(output_dim,))
    tf.reset_default_graph()
    config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8,
        device_count={'CPU': 8}
    )
    config.gpu_options.allow_growth = True
    sess = U.make_session(make_default=True, config=config)
    network = mlp(num_hidden=num_hidden, num_layers=num_layers)
    policy_train = build_policy(observation_space, action_space, network, trainable_variance=True,
                                trainable_bias=use_bias,
                                state_dependent_variance=True, beta=beta, init_logstd=init_logstd)()
    U.initialize()
    if model_path != '':
        policy_train.load(model_path)
    accuracy_nops = accuracy_left = accuracy_right = -1.
    if X is not None:
        states = np.array(X)
        actions = np.array(Y)
        nops_state = states[actions == 0]
        right_state = states[actions == 2]
        left_state = states[actions == 1]
        nops_action = actions[actions == 0]
        right_action = actions[actions == 2]
        left_action = actions[actions == 1]
        distribution = np.zeros(3)
        classes = np.unique(Y)
        class_counts = np.array([np.sum(Y == cl) for cl in classes])
        max_count = max(class_counts)
        for j,c in enumerate(classes):
            distribution[int(c)] = class_counts[j] / states.shape[0]
        accuracy_nops, _, loss, _ = policy_train.evaluate(nops_state, nops_action, False)
        print("Nops Accuracy:", accuracy_nops)
        if right_state.shape[0] > 0:
            accuracy_right, _, loss, _ = policy_train.evaluate(right_state, right_action, False)
        print("Right Accuracy:", accuracy_right)
        if left_state.shape[0] > 0:
            accuracy_left, _, loss, _ = policy_train.evaluate(left_state, left_action, False)
        print("Left Accuracy:", accuracy_left)
    return policy_train, [accuracy_nops, accuracy_left, accuracy_right], distribution


def evaluate_rule_based(pi, X, Y):
    accuracy_nops = accuracy_left = accuracy_right = 0
    states = np.array(X)
    actions = np.array(Y)
    nops_state = states[actions == 0]
    right_state = states[actions == 2]
    left_state = states[actions == 1]
    nops_action = actions[actions == 0]
    right_action = actions[actions == 2]
    left_action = actions[actions == 1]
    distribution = np.zeros(3)
    classes = np.unique(Y)
    class_counts = np.array([np.sum(Y == cl) for cl in classes])
    for j,c in enumerate(classes):
        distribution[int(c)] = class_counts[j] / states.shape[0]
    for i, s in enumerate(nops_state):
        act = pi(s)
        if act == nops_action[i]:
            accuracy_nops += 1
    accuracy_nops /= nops_state.shape[0]
    print("Nops Accuracy:", accuracy_nops)
    if right_state.shape[0] > 0:
        for i, s in enumerate(right_state):
            act = pi(s)
            if act == right_action[i]:
                accuracy_right += 1
        accuracy_right /= right_state.shape[0]
    else:
        accuracy_right = -1.
    print("Right Accuracy:", accuracy_right)
    if left_state.shape[0] > 0:
        for i, s in enumerate(left_state):
            act = pi(s)
            if act == left_action[i]:
                accuracy_left += 1
        accuracy_left /= left_state.shape[0]
    else:
        accuracy_left = -1.
    print("Left Accuracy:", accuracy_left)
    return pi, [accuracy_nops, accuracy_left, accuracy_right], distribution