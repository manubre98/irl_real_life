import numpy as np

# ttc_rb_th, ttc_rf_th, ttc_lb_th, ttc_lf_th, ego_speed_th, delta_v_th, distance_front_th
default_values = np.array(
    [[-150., 4.],
     [-150., 4.],
     [-150., 4.],
     [-150., 4.],
     [0.7, -1.],
     [0.2, -1.],
     [70., 2.]]
)


def extract_features(s, scaled=True):
    n_lanes = 3
    n_var = 4
    ego_speed = s[2]
    target_speed = 36.11
    max_distance = 110
    distance_front = s[n_var:n_var+n_lanes]
    distance_back = s[n_var+n_lanes:n_var+n_lanes*2]
    speed_front = s[n_var+n_lanes*2:n_var+n_lanes*3]
    speed_back = s[n_var+n_lanes*3:n_var+n_lanes*4]
    lanes = s[n_var+n_lanes*6:n_var+n_lanes*7]
    scale_factor_distance = (max_distance if scaled else 1)
    scale_factor_speed = (target_speed if scaled else 1)
    num_active_lanes = n_lanes
    changing_lane = s[3]

    if changing_lane:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, True

    lane_idx = np.argwhere(lanes == 1)[0][0]

    front_dist = distance_front[lane_idx] * scale_factor_distance
    distance_front = np.array(distance_front[:] * scale_factor_distance)
    speed_front = np.array(speed_front[:] * scale_factor_speed)
    distance_back = np.array(distance_back[:] * scale_factor_distance)
    speed_back = np.array(speed_back[:] * scale_factor_speed)
    ego_speed = ego_speed * scale_factor_speed
    ego_speed_relative = ego_speed / target_speed
    delta_v = (ego_speed - speed_front[lane_idx]) / target_speed

    if lane_idx == 0:
        ttc_right_back = np.inf
        ttc_right_front = np.inf
        gap_right = (0, 0)
    else:
        right_lane = lane_idx - 1
        gap_right = (distance_back[right_lane], distance_front[right_lane])
        ttc_right_front = get_ttc(ego_speed, speed_front[right_lane], distance_front[right_lane])
        ttc_right_back = get_ttc(speed_back[right_lane], ego_speed, distance_back[right_lane])

    if lane_idx == num_active_lanes - 1:
        ttc_left_back = np.inf
        ttc_left_front = np.inf
        gap_left = (0, 0)
    else:
        left_lane = lane_idx + 1
        gap_left = (distance_back[left_lane], distance_front[left_lane])
        ttc_left_front = get_ttc(ego_speed, speed_front[left_lane], distance_front[left_lane])
        ttc_left_back = get_ttc(speed_back[left_lane], ego_speed, distance_back[left_lane])

    return front_dist, ttc_right_back, ttc_right_front, ttc_left_back, ttc_left_front, \
           ego_speed_relative, delta_v, gap_right[0], gap_right[1], gap_left[0], gap_left[1], False


class RuleBasedPolicy:
    NUM_THETA = 7

    def __init__(self, init_values=None, baseline=0, alpha=0.1):

        if init_values is None:
            init_values = default_values
        self.epsilon = 1e-24
        self.rho = np.array(init_values, copy=True)
        self.theta = np.zeros(init_values.shape[0])
        self.baseline = baseline
        self.alpha = alpha
        self.old_return = 0
        self.t = 0
        self.resample()

    def resample(self):
        for i, th in enumerate(self.theta):
            self.theta[i] = np.random.normal(self.rho[i, 0], np.exp(self.rho[i, 1]))
        return np.copy(self.theta)

    def act(self, s):

        distance_front, ttc_right_back, ttc_right_front, ttc_left_back, ttc_left_front, ego_speed, delta_v, \
            gap_right_back, gap_right_front, gap_left_back, gap_left_front, changing_lane = s[:]

        ttc_rb_th, ttc_rf_th, ttc_lb_th, ttc_lf_th, ego_speed_th, delta_v_th, distance_front_th = self.theta[:]

        if changing_lane:
            return 0  # NOP
        if ttc_right_front < ttc_rf_th and ttc_right_back < ttc_rb_th and gap_right_back > 2.5 and gap_right_front > 12:
            return 2  # RIGHT
        elif ttc_left_front < ttc_lf_th and ttc_left_back < ttc_lb_th and distance_front < distance_front_th and \
                ego_speed < ego_speed_th and delta_v > delta_v_th and \
                gap_left_back > 2.5 and gap_left_front > 12:
            return 1  # LEFT
        else:
            return 0  # NOP

    def eval_params(self):
        return np.copy(self.rho)

    def get_theta(self):
        return np.copy(self.theta[:])

    def set_params(self, rho):
        self.rho[:, :] = rho[:, :]
        self.resample()

    def set_theta(self, theta):
        self.theta[:] = theta[:]

    def set_mean(self):
        self.theta[:] = self.rho[:, 0]

    def eval_gradient(self, thetas, returns, use_baseline=True):

        means = self.rho[:, 0]
        sigmas = np.exp(self.rho[:, 1])
        n = len(means)
        b = 0
        gradients = []
        gradient_norms = []
        self.t += 1
        for i, theta in enumerate(thetas):
            d_mu = (theta - means) / (sigmas**2 + self.epsilon)
            d_sigma = ((theta - means) ** 2 - sigmas ** 2) / (sigmas ** 3 + self.epsilon)
            gradients.append(np.concatenate((d_mu, d_sigma)))
            gradient_norms.append(np.linalg.norm(np.concatenate((d_mu, d_sigma))))
        if use_baseline:
            gradient_norms = np.array(gradient_norms)
            num = (returns * gradient_norms ** 2).mean()
            den = (gradient_norms ** 2).mean()
            b = num / den
        gradient = (gradients * (np.array(returns) - b)[:, np.newaxis]).mean(axis=0)
        d_mu = gradient[:n]
        d_sigma = gradient[n:]
        return np.stack([d_mu, d_sigma]).T

    def eval_natural_gradient(self, thetas, returns, use_baseline=True):

        means = self.rho[:, 0]
        sigmas = np.exp(self.rho[:, 1])
        n = len(means)
        b = 0
        gradients = []
        gradient_norms = []
        self.t += 1
        for i, theta in enumerate(thetas):
            d_mu = (theta - means)
            d_sigma = ((theta - means)**2 - sigmas**2) / (2 * sigmas**2 + self.epsilon)

            gradients.append(np.concatenate((d_mu, d_sigma)))
            gradient_norms.append(np.linalg.norm(np.concatenate((d_mu, d_sigma))))
        if use_baseline:
            gradient_norms = np.array(gradient_norms)
            num = (returns * gradient_norms ** 2).mean()
            den = (gradient_norms**2).mean()
            b = num / den
        gradient = (gradients * (np.array(returns) - b)[:, np.newaxis]).mean(axis=0)

        d_mu = gradient[:n]
        d_sigma = gradient[n:]

        return np.stack([d_mu, d_sigma]).T

    def get_baseline(self, ret):
        self.alpha = 1 / (self.t + 1)
        self.baseline = self.alpha * ret + (1-self.alpha) * self.baseline
        return self.baseline

    def show_theta(self):
        print(self.theta)

    @property
    def std(self):
        return np.exp(self.rho[:, 1])

    @property
    def mean(self):
        return self.rho[:, 0]

def get_ttc(follower_speed, leader_speed, leader_distance):
    const_dist = 6
    val = follower_speed**2 - leader_speed**2 + (follower_speed - leader_distance)*const_dist
    return val
