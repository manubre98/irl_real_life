import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as C
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import calendar
import os
from datetime import datetime, timedelta


LEGEND_FONT_SIZE = 22
AXIS_FONT_SIZE = 22
TICKS_FONT_SIZE = 16
LINE_WIDTH = 3.0


def plot_ts(data, attr, year_start, year_end=None):
    fig = plt.figure(figsize=(15, 8))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.set_xlabel('Time', fontsize=AXIS_FONT_SIZE)
    # axes.set_ylabel(attr + ' ' + str(dataset_number) + ' [msc] - ' + str(year_start), fontsize=AXIS_FONT_SIZE)
    axes.tick_params(labelsize=TICKS_FONT_SIZE)

    uniq, index = np.unique(data[(data['water_year'] == year_start)]["month"], return_index=True)
    uniq = uniq[index.argsort()]

    if year_end is None:

        axes.plot(range(1, 366, 1), data[(data['water_year'] == year_start)][attr].values)
        axes.set_xticks(np.arange(0, 360, 30))
        axes.set_xticklabels(uniq)

    else:

        cmap = plt.get_cmap('Blues')

        newcmap = matplotlib.colors.ListedColormap(cmap(np.linspace(0.35, 0.99, year_end - year_start)))

        for i, y in enumerate(range(year_start, year_end + 1)):
            axes.plot(range(1, 366, 1), data[(data['water_year'] == y)][attr], c=newcmap(i), alpha=1)

        norm = matplotlib.colors.Normalize(vmin=year_start, vmax=year_end)
        sm = plt.cm.ScalarMappable(cmap=newcmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ticks=np.linspace(year_start, year_end, year_end - year_start + 1),
                     boundaries=np.arange(year_start, year_end))
        axes.set_xticks(np.arange(0, 360, 30))
        axes.set_xticklabels(uniq)

    plt.show()

def plot_med_mean(data, attr, uniq, max_storage = None, q75=None, q25=None):
    #%matplotlib inline

    fig = plt.figure(figsize=(14, 8))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.set_xlabel('Time', fontsize=AXIS_FONT_SIZE)
    axes.set_ylabel('Mean ' + attr.replace("_", " "), fontsize=AXIS_FONT_SIZE)
    axes.tick_params(labelsize=TICKS_FONT_SIZE)

    start_year = data["water_year"].iloc[0]
    # end_year = start_year + 15
    end_year = data["water_year"].iloc[-1]

    for i, y in enumerate(range(start_year, end_year + 1)):
        axes.plot(data[(data['water_year'] == y)]["n_day_water"], data[(data['water_year'] == y)][attr], c="gray",
                  alpha=0.15)

    mean_ = data.groupby('n_day_water').mean()[attr].values
    median_ = data.groupby('n_day_water').median()[attr].values

    plt.plot(range(1, 366, 1), mean_, c="r", linewidth=3)
    plt.plot(range(1, 366, 1), median_, c="b", linewidth=3)
    if q75 is not None and q25 is not None:
        plt.fill_between(range(1, 366, 1), q25, q75, alpha=0.4)
        # plt.plot(range(1,366,1), q75, "--", c = "k", linewidth = 2)
        # plt.plot(range(1,366,1), q25, "--", c = "k", linewidth = 2)

    if attr == "storage":
        plt.plot(range(1, 366, 1), np.tile(max_storage, 365), "--", c="k", linewidth=3)

    plt.ylim(-1.1, 0.1)

    axes.set_xticks(np.arange(0, 360, 30))
    axes.set_xticklabels(uniq)

    plt.show()


def coeff_estimator(X, y, window_size):
    m = np.zeros(len(y))
    for i in range(len(y)):
        if i < (window_size):
            print(i)
            X_local = np.concatenate((np.tile(X.iloc[0], int(window_size - 1 - i)), X.iloc[:i + 1]))
            y_local = np.concatenate((np.tile(y.iloc[0], int(window_size - 1 - i)), y.iloc[:i + 1]))
        else:
            X_local = X.iloc[i - window_size + 1:i + 1].values
            y_local = y.iloc[i - window_size + 1:i + 1].values
        reg = LinearRegression().fit(X_local.reshape(-1, 1), y_local)
        m[i] = reg.coef_[0]
    return m


def mask_funct(inflection_1, inflection_2, max_q, left=0., center=1., right=0.):
    '''
    inflection_1: coordinate of first inflection point of the sigmoid
    inflection_2: coordinate of second inflection point of the sigmoid
    max_q: parameter to scale the sigmoid in order to be smooth with respect to the value assumed by the quantity
    increasing: boolean
    left: y-value of mask on the left side
    right: y-value of mask on the right side
       __
    __/  \__
    '''

    if center > left:
        increasing = True
    else:
        increasing = False

    a = 50 / max_q
    limit = max_q / 7

    width_l = abs(center - left)
    width_r = abs(center - right)

    if increasing:
        f_left = lambda x: (np.exp(a * (x - inflection_1)) / (1 + np.exp(a * (x - inflection_1)))) * width_l + left
        f_right = lambda x: (- np.exp(a * (x - inflection_2)) / (1 + np.exp(a * (x - inflection_2))) + 1) * width_r + right
    else:
        f_left = lambda x: (- np.exp(a * (x - inflection_1)) / (1 + np.exp(a * (x - inflection_1))) + 1) * width_l + center
        f_right = lambda x:(np.exp(a * (x - inflection_2)) / (1 + np.exp(a * (x - inflection_2)))) * width_r + center

    def func(x):
        sol = np.zeros_like(x, dtype=float)
        sol[x <= (inflection_1 + limit)] = f_left(x[x <= (inflection_1 + limit)])
        sol[x < (inflection_1 - limit)] = left
        sol[x > (inflection_1 + limit)] = f_right(x[x > (inflection_1 + limit)])
        sol[x > (inflection_2 + limit)] = right
        return sol

    return func


def sigmoid(inflection, max_q, left=0., right=1., increasing=True):
    '''
    inflection: coordinate of inflection point of the sigmoid
    max_q: parameter to scale the sigmoid in order to be smooth with respect to the value assumed by the quantity
    increasing: boolean
    left: y-value of sigmoid on the left side
    right: y-value of sigmoid on the right side
        ___
    ___/
    '''

    a = 50 / max_q
    limit = max_q / 5

    width = abs(left - right)

    if increasing:
        f = lambda x: (np.exp(a * np.clip(x - inflection, - limit, limit)) /
                       (1 + np.exp(a * np.clip(x - inflection, - limit, limit)))) * width + left
    else:
        f = lambda x: (- np.exp(a * np.clip(x - inflection, - limit, limit)) /
                       (1 + np.exp(a * np.clip(x - inflection, - limit, limit))) + 1) * width + right

    def func(x):
        return f(x)

    return func



def plot_med_mean_double(data, attr1, attr2, uniq, max_storage, q75=None, q25=None):
    #%matplotlib inline

    # plt.figure(figsize=(14,16))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))
    # axes= fig.add_axes([0.1,0.1,0.8,0.8])
    ax1.set_xlabel('Time', fontsize=AXIS_FONT_SIZE)
    ax1.set_ylabel('Mean ' + attr1.replace("_", " "), fontsize=AXIS_FONT_SIZE)
    ax1.tick_params(labelsize=TICKS_FONT_SIZE)

    start_year = data["water_year"].iloc[0]
    # end_year = start_year + 15
    end_year = data["water_year"].iloc[-1]

    for i, y in enumerate(range(start_year, end_year + 1)):
        ax1.plot(data[(data['water_year'] == y)]["n_day_water"], data[(data['water_year'] == y)][attr1], c="gray",
                 alpha=0.15)

    mean_ = data.groupby('n_day_water').mean()[attr1].values
    median_ = data.groupby('n_day_water').median()[attr1].values

    ax1.plot(range(1, 366, 1), mean_, c="r", linewidth=3)
    ax1.plot(range(1, 366, 1), median_, c="b", linewidth=3)
    if q75 is not None and q25 is not None:
        plt.fill_between(range(1, 366, 1), q25, q75, alpha=0.4)
        # plt.plot(range(1,366,1), q75, "--", c = "k", linewidth = 2)
        # plt.plot(range(1,366,1), q25, "--", c = "k", linewidth = 2)

    if attr1 == "storage":
        ax1.plot(range(1, 366, 1), np.tile(max_storage, 365), "--", c="k", linewidth=3)

    ax1.set_xticks(np.arange(0, 360, 30))
    ax1.set_xticklabels(uniq)

    ####

    ax2.set_xlabel('Time', fontsize=AXIS_FONT_SIZE)
    ax2.set_ylabel('Mean ' + attr2.replace("_", " "), fontsize=AXIS_FONT_SIZE)
    ax2.tick_params(labelsize=TICKS_FONT_SIZE)

    start_year = data["water_year"].iloc[0]
    # end_year = start_year + 15
    end_year = data["water_year"].iloc[-1]

    for i, y in enumerate(range(start_year, end_year + 1)):
        ax2.plot(data[(data['water_year'] == y)]["n_day_water"], data[(data['water_year'] == y)][attr2], c="gray",
                 alpha=0.15)

    mean_ = data.groupby('n_day_water').mean()[attr2].values
    median_ = data.groupby('n_day_water').median()[attr2].values

    ax2.plot(range(1, 366, 1), mean_, c="r", linewidth=3)
    ax2.plot(range(1, 366, 1), median_, c="b", linewidth=3)
    if q75 is not None and q25 is not None:
        plt.fill_between(range(1, 366, 1), q25, q75, alpha=0.4)
        # plt.plot(range(1,366,1), q75, "--", c = "k", linewidth = 2)
        # plt.plot(range(1,366,1), q25, "--", c = "k", linewidth = 2)

    if attr2 == "storage":
        plt.plot(range(1, 366, 1), np.tile(max_storage, 365), "--", c="k", linewidth=3)

    if attr2 == "outflow":
        plt.plot(range(1, 366, 1), np.tile(np.mean(median_), 365), "--", c="k", linewidth=3)

    ax2.set_xticks(np.arange(0, 360, 30))
    ax2.set_xticklabels(uniq)

    plt.show()


def preprocess(data):
    data = data[["date", "outflow", "storage", "water_year"]]
    data['date'] = pd.to_datetime(data['date'])
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data["hydro"] = data['storage'].values * data['outflow'].values

    data["n_day"] = np.zeros(len(data["date"]))

    for i in range(len(data["date"])):
        if calendar.isleap(data["date"].iloc[i].year) and data["date"].iloc[i].timetuple().tm_yday > 59:
            data["n_day"].iloc[i] = data["date"].iloc[i].timetuple().tm_yday - 1
        else:
            data["n_day"].iloc[i] = data["date"].iloc[i].timetuple().tm_yday

    data["n_day_water"] = data["n_day"]

    data.loc[(data['month'] == 10) | (data['month'] == 11) | (data['month'] == 12), "n_day_water"] -= 273
    data.loc[(data['month'] != 10) & (data['month'] != 11) & (data['month'] != 12), "n_day_water"] += (365 - 273)

    start_year = data['water_year'].iloc[0]

    uniq, index = np.unique(data[(data['water_year'] == start_year)]["month"], return_index=True)
    uniq = uniq[index.argsort()]

    outlier_data = data["outflow"].values
    window_size = 9

    window_mean = np.zeros_like(outlier_data)

    left_filler = [outlier_data[0]] * int(((window_size - 1) / 2))
    right_filler = [outlier_data[-1]] * int(((window_size - 1) / 2))

    outlier_data = np.concatenate((left_filler, outlier_data, right_filler))

    for i in range(len(window_mean)):
        window_mean[i] = np.nanmedian(outlier_data[i:i + window_size])

    data["outflow"] = window_mean

    res = window_mean - outlier_data[int(((window_size - 1) / 2)):-int(((window_size - 1) / 2))]
    res_nan = res[~np.isnan(res)]
    res_nan_not0 = res_nan[res_nan != 0]

    q_10 = np.quantile(res_nan_not0, 0.1)
    q_90 = np.quantile(res_nan_not0, 0.9)

    IQR = q_90 - q_10

    outlier_mask = np.zeros(len(res), dtype=bool)
    outlier_mask[abs(res) > 3 * IQR] = 1

    for i in range(len(outlier_mask)):
        if outlier_mask[i]:
            total_var = abs(data["storage"].iloc[i - 1] - data["storage"].iloc[i]) + abs(
                data["storage"].iloc[i + 1] - data["storage"].iloc[i])
            delta = abs(data["storage"].iloc[i - 1] - data["storage"].iloc[i + 1])

            if total_var > 3 * delta:
                data["storage"].iloc[i] = np.NaN

    data["storage"] = data["storage"].interpolate(method='spline', order=3)
    data["outflow"] = data["outflow"].interpolate(method='spline', order=3)

    return data, uniq


def feature_build(data, max_storage, action, global_mean, iqr_irr):
    m = coeff_estimator(X=data["n_day_water"], y=data["storage"], window_size=10)
    threshold = (max_storage / 365)
    median_ = data.groupby('n_day_water').median()["storage"].values
    max_outflow = np.quantile(data["outflow"], 0.99)

    ### IRRIGATION FEATURE

    # Mask to divide the first months from the last ones
    days = data["n_day"]
    max_ = np.argmax(data.groupby('n_day').median()["storage"].values)

    delta = np.max(median_) - np.min(median_)
    sig_delta = sigmoid(inflection=0.1 * max_storage, max_q=5 * max_storage, left=3, right=1 / 3, increasing=False)
    factor = sig_delta(delta)

    if max_ not in range(100, 280):
        max_ = 190

    start = max_ - 60
    end = max_ + 120
    if end > 365:
        end = 365
    f_outflow = mask_funct(inflection_1=start, inflection_2=end, max_q=365,
                           left=-1, center=1, right=-1)
    mask_year = f_outflow(days)

    windowed_data = data[(data["n_day"] > start) & (data["n_day"] < end)]
    max_outflow_window = np.max(windowed_data.groupby('n_day').median()["outflow"].values)

    windowed_data = data[~((data["n_day"] > start) & (data["n_day"] < end))]
    min_outflow_window = np.min(windowed_data.groupby('n_day').median()["outflow"].values)

    delta_outflow = max_outflow_window - min_outflow_window
    sig_delta_outflow = sigmoid(inflection=0.3 * max_outflow, max_q=5 * max_outflow, left=1, right=1 / 3,
                                increasing=False)
    factor_outflow = sig_delta_outflow(delta_outflow)

    ## MAIN COMPONENT
    feature_irr = (action - global_mean)
    feature_irr = feature_irr * mask_year

    ## MULTIPLICATIVE FACTOR
    # Increasing storage in 4-5-6
    start = max_ - 60
    end = max_
    f_day = mask_funct(inflection_1=start, inflection_2=end, max_q=365,
                       left=0, center=1, right=0)
    mask_first_months = f_day(days)

    sig_increasing_pos = sigmoid(inflection=threshold, max_q=0.1 * abs(np.max(m) - np.min(m)), left=1, right=2)
    mask_increasing_pos = sig_increasing_pos(m)

    sig_increasing_neg = sigmoid(inflection=threshold, max_q=0.1 * abs(np.max(m) - np.min(m)), left=1, right=1 / 4,
                                 increasing=False)
    mask_increasing_neg = sig_increasing_neg(m)

    mask_increasing_pos = np.ones(len(days)) * (1 - mask_first_months) + mask_increasing_pos * (mask_first_months)
    mask_increasing_neg = np.ones(len(days)) * (1 - mask_first_months) + mask_increasing_neg * (mask_first_months)

    ## MULTIPLICATIVE FACTOR
    # Decreasing storage in 7-8-9
    start = max_
    end = max_ + 120
    if end > 365:
        end = 365
    f_day = mask_funct(inflection_1=start, inflection_2=end, max_q=365,
                       left=0, center=1, right=0)
    mask_last_months = f_day(days)

    sig_decreasing_pos = sigmoid(inflection=- threshold, max_q=abs(np.max(m) - np.min(m)), left=2, right=1,
                                 increasing=False)
    mask_decreasing_pos = sig_decreasing_pos(m)

    sig_decreasing_neg = sigmoid(inflection=- threshold, max_q=abs(np.max(m) - np.min(m)), left=1 / 4, right=1,
                                 increasing=True)
    mask_decreasing_neg = sig_decreasing_neg(m)

    mask_decreasing_pos = np.ones(len(days)) * (1 - mask_last_months) + mask_decreasing_pos * (mask_last_months)
    mask_decreasing_neg = np.ones(len(days)) * (1 - mask_last_months) + mask_decreasing_neg * (mask_last_months)

    feature_irr[feature_irr > 0] = feature_irr[feature_irr > 0] * mask_increasing_pos[feature_irr > 0] * \
                                   mask_decreasing_pos[feature_irr > 0] * factor
    feature_irr[feature_irr < 0] = feature_irr[feature_irr < 0] * mask_increasing_neg[feature_irr < 0] * \
                                   mask_decreasing_neg[feature_irr < 0] * factor * factor_outflow

    # Mean to preserve smoothness
    old_feature_irr = np.concatenate(([feature_irr[0]], feature_irr[:-1]))
    old_old_feature_irr = np.concatenate(([old_feature_irr[0]], old_feature_irr[:-1]))
    feature_irr = 1 / 3 * (old_old_feature_irr + old_feature_irr + feature_irr)

    ## HYDRO

    sig_hydro = sigmoid(inflection=0.05 * max_storage, max_q=1 / 2 * max_outflow, left=1 / 4, right=1)
    factor_hydro = sig_hydro(np.max(iqr_irr))

    modest_outflow = 0.1 * max_outflow

    # Mask to increase reward if water storage is HIGH
    sig_storage = sigmoid(inflection=0.7 * max_storage, max_q=max_storage, left=1, right=2)
    mask_storage = sig_storage(data["storage"])

    # Mask to decrease reward if water storage is LOW
    sig_storage_low_neg = sigmoid(inflection=0.5 * max_storage, max_q=5 * max_storage, left=2, right=1,
                                  increasing=False)
    mask_storage_low_neg = sig_storage_low_neg(data["storage"])

    # Mask to increase reward if water storage is CONSTANT
    f_const = mask_funct(inflection_1=- threshold, inflection_2=threshold, max_q=0.1 * abs(np.max(m) - np.min(m)),
                         left=0.5, center=1.5, right=0.5)
    mask_const = f_const(m)

    feature_hydro = (action - modest_outflow)
    # mask_storage_low_neg = np.ones(len(days)) * (1 - mask_const) + mask_storage_low_neg * (mask_const)

    feature_hydro[feature_hydro > 0] = feature_hydro[feature_hydro > 0] * mask_const[feature_hydro > 0] * mask_storage[
        feature_hydro > 0] * factor_hydro

    feature_hydro[feature_hydro < 0] = feature_hydro[feature_hydro < 0] * mask_storage_low_neg[
        feature_hydro < 0] * factor_hydro

    # Mean to preserve smoothness
    old_feature_hydro = np.concatenate(([feature_hydro[0]], feature_hydro[:-1]))
    old_old_feature_hydro = np.concatenate(([old_feature_hydro[0]], old_feature_hydro[:-1]))
    feature_hydro = 1 / 3 * (old_old_feature_hydro + old_feature_hydro + feature_hydro)

    ## FLOOD

    # Penalize if you don't produce enough energy
    scaling_factor = np.sqrt(np.mean(data["diff_med_mean"]))
    if scaling_factor > 1:
        modest_outflow_flood = modest_outflow / scaling_factor
    else:
        modest_outflow_flood = modest_outflow

    # Mask to decrease reward if water storage is HIGH
    sig_storage_high = sigmoid(inflection=0.5 * max_storage, max_q=max_storage, left=1, right=-1, increasing=False)
    mask_storage_high = sig_storage_high(data["storage"])

    # Mask to increase reward if water storage is LOW
    sig_storage_low_pos = sigmoid(inflection=0.3 * max_storage, max_q=max_storage, left=2, right=1, increasing=False)
    mask_storage_low_pos = sig_storage_low_pos(data["storage"])

    # Mask to increase reward if water storage is LOW
    sig_storage_low_neg = sigmoid(inflection=0.3 * max_storage, max_q=5 * max_storage, left=1 / 2, right=1,
                                  increasing=True)
    mask_storage_low_neg = sig_storage_low_neg(data["storage"])

    # Mask to see if water storage is CONSTANT
    f_const = mask_funct(inflection_1=- threshold, inflection_2=threshold, max_q=abs(np.max(m) - np.min(m)),
                         left=0, center=1, right=0)
    mask_const = f_const(m)

    # Mask to PENALIZE if water storage is VERY HIGH
    sig_storage_very_high = sigmoid(inflection=0.9 * max_storage, max_q=max_storage, left=1, right=- 0.2,
                                    increasing=False)
    mask_storage_very_high = sig_storage_very_high(data["storage"])

    # Mask to increase reward if mean is greater than median
    sig_mean_med = sigmoid(inflection=0.2 * max(data["diff_med_mean"]), max_q=max(data["diff_med_mean"]), left=1,
                           right=2)
    mask_mean_med = sig_mean_med(data["storage"])

    mask_storage_high = np.ones(len(days)) * (1 - mask_const) + mask_storage_high * (mask_const)
    mask_storage_low_pos = np.ones(len(days)) * (1 - mask_const) + mask_storage_low_pos * (mask_const)
    mask_storage_low_neg = np.ones(len(days)) * (1 - mask_const) + mask_storage_low_neg * (mask_const)

    # mask_storage_low = mask_storage_low - (mask_storage_low - 1) * (1 - mask_const)

    feature_flood = (action - modest_outflow_flood)

    feature_flood[feature_flood > 0] = feature_flood[feature_flood > 0] * mask_storage_high[feature_flood > 0]
    feature_flood[feature_flood > 0] = feature_flood[feature_flood > 0] * mask_storage_low_pos[feature_flood > 0]
    feature_flood[feature_flood < 0] = feature_flood[feature_flood < 0] * mask_storage_low_neg[feature_flood < 0]

    feature_flood[feature_flood > 0] = feature_flood[feature_flood > 0] * mask_storage_very_high[feature_flood > 0]
    feature_flood[feature_flood > 0] = feature_flood[feature_flood > 0] * mask_mean_med[feature_flood > 0]

    # Mean to preserve smoothness
    old_feature_flood = np.concatenate(([feature_flood[0]], feature_flood[:-1]))
    old_old_feature_flood = np.concatenate(([old_feature_flood[0]], old_feature_flood[:-1]))
    feature_flood = 1 / 3 * (old_old_feature_flood + old_feature_flood + feature_flood)

    ## FEATURE AGGREGATION & NORMALIZATION

    data["rew_irr"] = feature_irr
    data["rew_hydro"] = feature_hydro
    data["rew_flood"] = feature_flood

    mean_irr = data.groupby('n_day_water').mean()["rew_irr"].values
    median_irr = data.groupby('n_day_water').median()["rew_irr"].values

    mean_hydro = data.groupby('n_day_water').mean()["rew_hydro"].values
    median_hydro = data.groupby('n_day_water').median()["rew_hydro"].values

    mean_flood = data.groupby('n_day_water').mean()["rew_flood"].values
    median_flood = data.groupby('n_day_water').median()["rew_flood"].values

    max_feat = max(max(median_irr), max(median_hydro), max(median_flood))
    min_feat = min(min(median_irr), min(median_hydro), min(median_flood))

    all_rewards = np.concatenate((feature_irr, feature_hydro, feature_flood))
    inflection = np.quantile(all_rewards[all_rewards > 0], 0.7)
    sig_normalize = sigmoid(inflection=inflection, max_q=5 * max_feat, left=0, right=0.01)

    feature_irr[feature_irr > 0] = 0
    feature_hydro[feature_hydro > 0] = 0
    feature_flood[feature_flood > 0] = 0

    feature_irr[feature_irr < 0] = feature_irr[feature_irr < 0] / np.abs(min_feat)
    feature_hydro[feature_hydro < 0] = feature_hydro[feature_hydro < 0] / np.abs(min_feat)
    feature_flood[feature_flood < 0] = feature_flood[feature_flood < 0] / np.abs(min_feat)

    feature_irr[feature_irr < -1] = -1
    feature_hydro[feature_hydro < -1] = -1
    feature_flood[feature_flood < -1] = -1

    feature_irr[feature_irr < np.quantile(feature_irr[feature_irr < 0], 0.05)] = 0
    try:
        feature_hydro[feature_hydro < np.quantile(feature_hydro[feature_hydro < 0], 0.05)] = 0
    except IndexError:
        pass
    try:
        feature_flood[feature_flood < np.quantile(feature_flood[feature_flood < 0], 0.05)] = 0
    except IndexError:
        pass

    feature_hydro = feature_hydro[:, None]
    feature_flood = feature_flood[:, None]
    feature_irr = feature_irr[:, None]

    reward_features = np.hstack([feature_irr, feature_hydro, feature_flood])
    return reward_features
