import os
import sys
import tensorflow as tf
import numpy as np
import math
from tqdm import trange
import random
import time
from gym.spaces import Box, Discrete
from policies.linear_gaussian_policy import LinearGaussianPolicy
path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path
from baselines.common.policies_bc import build_policy
from baselines.common.models import mlp
import baselines.common.tf_util as U
from sklearn.linear_model import LinearRegression


def behavioral_cloning_nn(num_epochs, num_layers, num_hidden, X, Y, validation =0.2, lr=1e-4, l2=0., batch_size=128,
                          init_logstd=1., state_dependent_variance=True, starting_point='',
                          discrete=False, beta=1.0):
    input_dim = X.shape[-1]
    output_dim = Y.shape[-1]
    observation_space = Box(low=-np.inf, high=np.inf, shape=(input_dim,))
    if discrete:
        action_space = Discrete(n=len(np.unique(Y)))
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
    policy_train = build_policy(observation_space, action_space, network, l2=l2, lr=lr,
                                trainable_variance=state_dependent_variance, init_logstd=init_logstd, beta=beta,
                                state_dependent_variance=state_dependent_variance)()
    U.initialize()
    if starting_point != '':
        policy_train.load(starting_point)
    # dataset build
    states = X
    actions = Y

    if discrete:
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
    dataset = list(zip(states, actions))
    random.shuffle(dataset)
    if validation > 0.:
        k = math.floor(validation * len(dataset))
        dataset_training = dataset[:-k]
        dataset_validation = dataset[-k:]
    else:
        dataset_training = dataset[:]

    # pre-processing statistics
    num_batches = len(dataset_training) // batch_size
    num_batches += (0 if len(dataset_training) % batch_size == 0 else 1)
    print('# batches: ', num_batches)
    print('# training samples: ', len(dataset_training))
    logger = {
        'training_samples': len(dataset_training),
        'batch_size': batch_size,
        'num_batches': num_batches,
        'num_epochs': num_epochs
    }
    if validation > 0.:
        print('# validation samples: ', len(dataset_validation))
        logger['validation_samples'] = len(dataset_validation)

        # validation samples built
        X_val, y_val = zip(*dataset_validation)
        X_val, y_val = np.array(X_val), np.array(y_val)
    # train + accuracy over epochs
    counter = 0
    best_loss = np.inf
    for epoch in trange(num_epochs):
        # train batches built
        random.shuffle(dataset_training)
        batches = []
        for i in range(num_batches):
            base = batch_size * i
            batches.append(dataset_training[base: base + batch_size])
        # train
        if validation > 0.:
            target = y_val
            accuracy, _, loss = policy_train.evaluate(X_val[:], target, False)
            if epoch % 1 == 0 and loss <= best_loss:
                best_loss = loss
        else:
            pass
        for batch in batches:

            batch_X, batch_y = zip(*batch)
            target = batch_y
            output = policy_train.fit(batch_X, target)
            summaries = [tf.Summary.Value(tag="loss", simple_value=output[0]),
                         tf.Summary.Value(tag="r2", simple_value=output[1])]
            if not discrete:
                summaries += [tf.Summary.Value(tag="mean_std", simple_value=output[2]),
                              tf.Summary.Value(tag="min_std", simple_value=output[3]),
                              tf.Summary.Value(tag="max_std", simple_value=output[4])]
            else:
                summaries += [tf.Summary.Value(tag="entropy", simple_value=output[2]),
                              tf.Summary.Value(tag="stochastic_accuracy", simple_value=output[3])]
            counter += 1
        # validation
    if validation > 0.:
        target = y_val
        accuracy, _, loss = policy_train.evaluate(X_val[:], target, False)
        summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=accuracy),
                                    tf.Summary.Value(tag="test_loss", simple_value=loss)])
        if num_epochs % 1 == 0 and loss <= best_loss:
            best_loss = loss
    batch_X, batch_Y = zip(*dataset)
    _, _, loss, ll = policy_train.evaluate(batch_X[:], batch_Y[:], False)
    logger['cost'] = loss
    logger['ll'] = ll
    return policy_train, logger, None


def behavioral_clonning_linear(X, Y, noise=None, name=''):
    regr = LinearRegression()
    train_data = X
    train_label = Y
    regr.fit(train_data, train_label)
    pi = LinearGaussianPolicy(weights=regr.coef_, bias=regr.intercept_, noise=noise)
    model_name = 'linear'
    model_path = 'logs/tensorboards/' + name + '/' + model_name + '_' + str(time.time()) + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    np.save(model_path + 'best.npy', regr.coef_)

    predictions = regr.predict(train_data)
    cost = np.sum((predictions - train_label) ** 2)

    std = 1
    ll = -np.sum(- 0.5 * np.log(2 * np.pi) - np.log(std) - 0.5 * (predictions - train_label) ** 2 / (
                std ** 2 + 1e-10))

    #print("R2 Score:" + str(regr.score(X, Y)))

    logger = {
        'training_samples': len(train_data),
        'cost': cost,
        'll': ll
    }
    return pi, logger, model_path


def load_policy(model_path, input_dim, output_dim, num_hidden, num_layers, init_logstd=1., discrete=False,
                beta=1.0):
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
                                state_dependent_variance=True, beta=beta, init_logstd=init_logstd)()
    U.initialize()
    policy_train.load(model_path)
    return policy_train