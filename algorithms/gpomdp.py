import numpy as np

class GPOMDP():
    def __init__(self, gradient, gamma, rewards, weights=None, probs=[1]):
        self.gradients = gradient
        self.gamma = gamma
        self.weights = weights
        self.rewards = rewards
        self.probs = probs

    def eval_gpomdp(self, normalize_features=False):
        if self.weights == None:
            self.weights = 1
        if len(self.probs) == 1:
            self.probs = np.ones((self.gradients.shape))
        sn = (np.mean(self.probs, axis=0) + 1.e-10)
        discount_factor_timestep = np.power(self.gamma * np.ones(self.gradients.shape[1]),
                                                range(self.gradients.shape[1]))  # (T,)
        print(discount_factor_timestep.shape)
        discounted_return = discount_factor_timestep[np.newaxis, :, np.newaxis] * self.rewards# (N,T,L)
        print("Feature Expectations:")
        print(discounted_return.sum(axis=1).mean(axis=0))
        if normalize_features:
            discounted_return /= discounted_return.sum(axis=1).mean(axis=0)
        print(discounted_return.shape)
        # print(discounted_return.cumsum(axis=1), discounted_return.cumsum(axis=1).shape)
        gradient_est_timestep = np.cumsum(self.gradients, axis=1) # (N,T,K)
        print(gradient_est_timestep.shape)
        gradient_est_timestep2 = np.cumsum(self.gradients, axis=1) ** 2  # (N,T,K)
        print(gradient_est_timestep2.shape)
        baseline_den = np.mean(gradient_est_timestep2, axis=0)  # (T,K)
        print(baseline_den.shape)
        baseline_num = np.mean(
            (gradient_est_timestep2)[:, :, :, np.newaxis] * discounted_return[:, :, np.newaxis, :],
            axis=0)  # (T,K,L)
        print(baseline_num.shape)
        baseline = baseline_num / (baseline_den[:, :, np.newaxis] + 1e-7)  # (T,K,L)
        print(baseline.shape)
        gradient = np.sum(gradient_est_timestep[:, :, :, np.newaxis] * (discounted_return[:, :, np.newaxis, :] -
                                                                    baseline[np.newaxis, :]), axis=1)  # (N,K,L)
        print(gradient.shape)
        return gradient

    def eval_gpomdp_discrete(self):
        if self.weights == None:
            self.weights = 1
        if len(self.probs) == 1:
            self.probs = np.ones((self.gradients.shape[0], self.gradients.shape[1]))
        discount_factor_timestep = np.power(self.gamma * np.ones(self.gradients.shape[1]),
                                                range(self.gradients.shape[1]))  # (T,)
        discounted_return = discount_factor_timestep[np.newaxis, :, np.newaxis] * self.rewards# (N,T,L)
        gradient_est_timestep = np.cumsum(self.gradients, axis=1) * self.probs[:,:,np.newaxis]  # (N,T,K, 2)
        baseline_den = np.mean(gradient_est_timestep ** 2 + 1.e-10, axis=0)  # (T,K, 2)
        baseline_num = np.mean(
            (gradient_est_timestep ** 2)[:, :, :, np.newaxis] * discounted_return[:, :, np.newaxis, :],
            axis=0)  # (T,K,2,L)
        baseline = baseline_num / baseline_den[:, :, np.newaxis]  # (T,K,2,L)
        gradient = np.mean(
            np.sum(gradient_est_timestep[:, :, :, np.newaxis] * (discounted_return[:, :, np.newaxis, :] -
                                                                    baseline[np.newaxis, :, :]), axis=1),
            axis=0)  # (K,2,L)
        return gradient