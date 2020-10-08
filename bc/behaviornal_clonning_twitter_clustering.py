import os
import sys
import tensorflow as tf
import numpy as np
import math
from tqdm import trange
import random
from gym.spaces import Box, Discrete
path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path
from baselines.common.policies_bc import build_policy
from baselines.common.models import mlp
import baselines.common.tf_util as U

ACTIONS = ['nop', 'tweet']


def bc_twitter(states, actions, args, tf_path, model_path):
    tweet_states = states[actions == 1]
    tweet_actions = actions[actions == 1]
    nop_states = states[actions == 0]
    nop_actions = actions[actions == 0]
    num_tweet = (actions == 1).sum()
    num_nop = (actions == 0).sum()
    if num_nop > args.ratio * num_tweet:
        for i in range(min(int((num_nop / num_tweet) // 2), 10)):
            states = np.concatenate((states, tweet_states))
            actions = np.concatenate((actions, tweet_actions))
    elif num_tweet > args.ratio * num_nop:
        for i in range(min(int((num_tweet / num_nop) // 2), 10)):
            states = np.concatenate((states, nop_states))
            actions = np.concatenate((actions, nop_actions))
    num_tweet = (actions == 1).sum()
    num_nop = (actions == 0).sum()
    if args.weight_classes:
        ratio = num_tweet / (num_tweet + num_nop)
        class_weights = [ratio, 1 - ratio]
    else:
        class_weights = None
    dataset = list(zip(states, actions))
    random.shuffle(dataset)

    observation_space = Box(low=-np.inf, high=np.inf, shape=(len(states[0]),))
    action_space = Discrete(len(ACTIONS))
    tf.reset_default_graph()
    sess = U.make_session(make_default=True)
    network = mlp(num_hidden=args.num_hidden, num_layers=args.num_layers)
    policy_train = build_policy(observation_space, action_space, network, l2=args.l2, lr=args.lr, train=True,
                                class_weights=class_weights)()
    U.initialize()
    writer = tf.summary.FileWriter(tf_path)

    if args.validation > 0.:
        k = math.floor(args.validation * len(dataset))
        dataset_training = dataset[:-k]
        dataset_validation = dataset[-k:]
    else:
        dataset_training = dataset[:]

    # pre-processing statistics
    num_batches = len(dataset_training) // args.batch_size
    if len(dataset_training) % args.batch_size > 0:
        num_batches += 1
    print('# batches: ', num_batches)
    print('# training samples: ', len(dataset_training))
    logger = {
        'training_samples': len(dataset_training),
        'batch_size': args.batch_size,
        'num_batches': num_batches,
        'num_epochs': args.num_epochs
    }
    if args.validation > 0.:
        print('# validation samples: ', len(dataset_validation))
        logger['validation_samples'] = len(dataset_validation)

    # validation samples built
    X_val, y_val = zip(*dataset_validation)
    X_val, y_val = np.array(X_val), np.array(y_val)
    XX_val, yy_val = [], []
    for i in range(len(ACTIONS)):
        XX_val.append(X_val[y_val == i])
        yy_val.append(y_val[y_val == i])

    # train + accuracy over epochs
    counter = 0
    best_accuracy = 0
    for epoch in trange(args.num_epochs):
        # train batches built
        random.shuffle(dataset_training)
        batches = []
        for i in range(num_batches):
            base = args.batch_size * i
            batches.append(dataset_training[base: base + args.batch_size])
        # train
        try:
            for batch in batches:
                batch_X, batch_y = zip(*batch)
                output = policy_train.fit(batch_X, batch_y)
                summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=output[0]), ])
                writer.add_summary(summary, counter)
                counter += 1
        except:
            print("Error")
        # validation
        if args.validation > 0.:
            overall_accuracy = 0
            for i in range(len(ACTIONS)):
                try:
                    accuracy, _, _, _ = policy_train.evaluate(XX_val[i], yy_val[i], args.stochastic_eval)
                except Exception as e:
                    print(e)
                summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy_" + ACTIONS[i], simple_value=accuracy), ])
                writer.add_summary(summary, epoch)
                overall_accuracy += accuracy
            overall_accuracy /= len(ACTIONS)
            summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy_overall", simple_value=overall_accuracy), ])
            writer.add_summary(summary, epoch)
            if epoch % 10 == 0 and best_accuracy <= overall_accuracy:
                policy_train.save(model_path + 'best')
                best_accuracy = overall_accuracy
    with open(tf_path + '/log.txt', 'w') as log_file:
        log_file.write(str(logger))

    return policy_train, sess
