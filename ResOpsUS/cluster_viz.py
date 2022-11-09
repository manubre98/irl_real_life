import pickle

file = open("/Users/Manuel/Desktop/irl_real_life/ResOpsUS/Results/results_new_one_3.pkl", "rb")

cluster = pickle.load(file)

file.close()