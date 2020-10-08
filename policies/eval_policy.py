import numpy as np
import time


def eval_policy(env, pi, n_episodes, verbose=True, interactive=False):

    rewards = []
    logs = []
    feature_expetations = []
    for i in range(n_episodes):

        start = time.time()
        s = env.reset()
        t = 0
        rew = 0
        features = []
        while True:
            a = pi(s)
            ns, r, done, inf = env.step(a[0])
            s = ns
            if interactive:
                print("Action=%d" % a)
                print("Reward=%f" % r)
                input()
            rew += r
            t += 1
            features.append(inf['features'])
            if done:
                break

        if verbose:
            print("Episode {0}: Return = {1}, Duration = {2}, Time = {3} s".format(i, rew, t, time.time() - start))
        features = np.array(features)
        feature_expetations.append(np.sum(features, axis=0))
        rewards.append(rew)
        logs.append({"reward": rew})
    features = np.mean(feature_expetations, axis=0)
    logs.append({'feature_expectations':features })
    avg = np.mean(rewards)
    std = np.std(rewards)
    if verbose:
        print("Average Return = {0} +- {1}".format(avg, std))
        print("Feature Expectations = Too High:{0} \t Demand:{1} \t Too Low:{2}".format(features[0], features[1],
                                                                                        features[2]))
    env.reset()

    return avg, std, logs
