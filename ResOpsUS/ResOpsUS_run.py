import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
from datetime import datetime, timedelta
from utils import preprocess, feature_build
import warnings
import subprocess
import sys

#warnings.filterwarnings('ignore')

dataset_list = [993]
for dataset_number in dataset_list:
    base_dir = '../ResOpsUS/data/' + str(dataset_number) + "/"
    logs_dir = '../logs/ResOpsUS/' + str(dataset_number) + "/"

    print(dataset_number)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    data = pd.read_csv('../ResOpsUS/time_series/' + "ResOpsUS_" + str(dataset_number) + ".csv", delimiter=',')

    G = pd.read_excel("../ResOpsUS/GRanD_dams_v1_3.xls")
    max_storage = int(G.iloc[dataset_number - 1]["NEW_STORAGE"])

    data, uniq = preprocess(data)

    min_max_scaler = MinMaxScaler()
    n_year = len(np.unique(data['water_year']))

    # Summary Statistics
    mean_outflow = data.groupby('n_day_water').mean()["outflow"].values
    median_outflow = data.groupby('n_day_water').median()["outflow"].values
    global_mean = np.mean(median_outflow)
    data["mean_outflow"] = np.tile(mean_outflow, int(data.shape[0] / 365))
    data["median_outflow"] = np.tile(median_outflow, int(data.shape[0] / 365))
    data["global_mean"] = np.tile(global_mean, n_year * 365)
    data["diff_med_mean"] = data["mean_outflow"] - data["median_outflow"]

    # Actions
    action = data['outflow'].values.ravel()

    # Day and Month
    months = pd.get_dummies(data['month']).values
    day = data['n_day_water'].values.ravel()

    # State features
    storage = data['storage'].values.ravel()
    old1_action = np.concatenate(([action[0]], action[:-1]))
    old2_action = np.concatenate(([old1_action[0]], old1_action[:-1]))
    old3_action = np.concatenate(([old2_action[0]], old2_action[:-1]))
    old4_action = np.concatenate(([old3_action[0]], old3_action[:-1]))
    old5_action = np.concatenate(([old4_action[0]], old4_action[:-1]))

    state = np.array([storage, old1_action, old2_action,
                      old3_action, old4_action, old5_action]).T

    state = np.concatenate((state, months[:,:-1]), axis=1)

    # Storage
    iqr_stor = np.zeros(365)
    q90_stor = np.zeros(365)
    q10_stor = np.zeros(365)

    for i in range(365):
        q90_stor[i], q10_stor[i] = np.percentile(storage.reshape(-1, 365)[:, i], [90, 10])
        iqr_stor[i] = q90_stor[i] - q10_stor[i]

    iqr_stor = np.tile(iqr_stor, int(len(action) / 365)) + 1e-5

    reward_features = feature_build(data, max_storage, action, global_mean, iqr_stor)

    #scale states
    state_before_scaling = state
    state = min_max_scaler.fit_transform(state)
    action_before_scaling = action
    action = min_max_scaler.fit_transform(action[:, None])

    np.save(base_dir + 'states.npy', state)
    np.save(base_dir + 'actions.npy', action_before_scaling[:, None])
    np.save(base_dir + 'reward_features.npy', reward_features)

    subprocess.run([sys.executable,
                    '/Users/Manuel/Desktop/irl_real_life/run_irl_usa.py',
                    '--dataset_number', str(dataset_number)])

