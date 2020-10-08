import os
import numpy as np
import pandas as pd


def convert(string):
    raw_elements = string[1:-1].split()
    return [float(x) for x in raw_elements]


def read_trajectories(logs, next_states_flag=False, fix_safety_violations=False):
    files = os.listdir(logs)
    dataset_states = []
    dataset_actions = []
    dataset_rewards = []
    dataset_next_states = []
    dataset_dones = []
    for file in files:
        if '.csv' not in file or 'lock' in file:
            continue
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        df = pd.read_csv(logs + file)
        try:
            df_current = df.loc[:, :'ego_lanes']
            df_next = df.loc[:, "free_right'":"ego_lanes'"]
            df_action = df['action']
            df_done = df['done']
        except:
            continue
        for index, row in df_current.iterrows():
            #read state
            state = []
            for j, col in enumerate(row):
                if j > 3:
                    if 4 <= j <= 7:
                        state += (np.array(convert(col))).tolist()
                    else:
                        state += convert(col)
                else:
                    state.append(float(col))

            #read next state
            next_state = []
            row_next = df_next.loc[index, :]
            for j, col in enumerate(row_next):
                if j > 3:
                    if 4 <= j <= 7:
                        next_state += (np.array(convert(col))).tolist()
                    else:
                        next_state += convert(col)
                else:
                    next_state.append(float(col))

            #read action
            action = int(df_action[index])  # [1])

            #read reward_features
            # df_reward = df.loc[index, 'r_deltaspeed':'r_jerk']

            df_reward = df.loc[index, ['r_freeright', 'r_lanechange', 'r_safetyviolation', 'r_distancefront']]
            reward = [float(r) for r in df_reward]
            if fix_safety_violations:
                if reward[2] != 0:
                    reward[2] = 0
                    action = 0
                if action != 0 and reward[1] != -1: # check
                    print("Whaat")
            ego_lane = np.array(convert(df.loc[index, "ego_lanes'"]))
            lane_index = np.argmax(ego_lane)
            front_distance = np.array(convert(df.loc[index, "distance_front'"]))
            vehicle_in_front = front_distance[lane_index]

            if not np.isclose(reward[-1], vehicle_in_front - 1): # check
                print("Whattt")
            if action != 0:
                reward[1] *= 31  # lane change has high value
            #read done
            done = df_done[index] == "True"
            reward = np.array(reward)[[0, 1, 3]].tolist()
            #add to dataset
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        dataset_states += states
        dataset_actions += actions
        dataset_rewards += rewards
        dataset_next_states += next_states
        dataset_dones += dones
    if next_states_flag:
        return dataset_states, dataset_actions, dataset_rewards, dataset_next_states, dataset_dones
    else:
        return dataset_states, dataset_actions, dataset_rewards


if __name__ == '__main__':
    agents = ['Bob', 'Alice', 'Carol', 'Chuck', 'Craig', 'Dan', 'Erin', 'Eve', 'Grace', 'Judy']
    base_dir = '../datasets/highway/'
    for agent in agents:
        logs = base_dir + agent + '/'
        states, actions, rewards = read_trajectories(logs + 'logs/', fix_safety_violations=True)
        np.save(logs + 'states.npy', states)
        np.save(logs + 'actions.npy', actions)
        np.save(logs + 'rewards.npy', rewards)
        print("Done Agent:", agent)

