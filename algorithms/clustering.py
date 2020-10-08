import numpy as np
from algorithms.sigma_girl import make_loss_function, make_weights_assignment_function
from utils.utils import  algorithm_u


def em_clustering(mus, sigmas, ids, lamb=1, num_clusters=2, num_objectives=3, max_iterations=100,
                  verbose=False, optimization_iterations=10, girl=False):
    num_agents = len(mus)
    loss_functions = []
    tolerance = 1e-3
    for i, mu in enumerate(mus):
        loss_functions.append(make_loss_function(mu, sigmas[i], ids[i], girl=girl))

    weight_calculator = make_weights_assignment_function(mus, sigmas, ids, num_objectives=num_objectives,
                                                         verbose=verbose,
                                                         num_iters=optimization_iterations, girl=girl)
    min_loss = np.inf
    best_assignment = None

    # initial assignment

    p = np.zeros((num_clusters, num_agents))
    for i in range(num_clusters):
        c = np.random.choice(num_clusters)
        p[c, i] = 1.
    # Best Assignment
    omega = np.zeros(shape=(num_clusters, num_objectives))
    it = 0
    omega = np.random.random((num_clusters, num_objectives))
    prev_assignment = None

    while it < max_iterations:
        it += 1
        # find best omega for assignment
        # equal to minimizing a function for each separate cluster
        for i in range(num_clusters):
            w, loss = weight_calculator(p[i])
            omega[i] = w
        if verbose:
            print("Omegas at iteration %d: " % it)
            print(omega)

        # find best assignment for given Omega
        # best assignment is hard
        loss_value = 0
        for j, mu in enumerate(mus):
            p[:, j] = 0.
            minimum = np.inf
            # index = -1
            for i in range(num_clusters):
                loss = loss_functions[j](omega[i])
                if loss < minimum:
                    # index = i
                    minimum = loss
                p[i, j] = -lamb * loss
            p[:, j] -= np.max(p[:, j])
            p[:, j] = np.exp(p[:, j])
            p[:, j] /= np.sum(p[:, j])
            loss_value += minimum
        if verbose:
            print("Assignment at iteration %d: " % it)
            print(p)
            print("Loss Value ", loss_value)
        if loss_value < min_loss:
            min_loss = loss_value
            best_assignment = np.copy(p)
        if prev_assignment is not None:
            diff = np.max(np.abs(prev_assignment - p))
            if diff < tolerance:
                if verbose:
                    print("Converged at Iteration %d:" % it)
                break
        # perturbation of assignment
        prev_assignment = np.copy(p)
    if it == max_iterations:
        if verbose:
            print("Finished %d iterations without converging" % max_iterations)
    for i in range(num_agents):
        best = np.argmax(best_assignment[:, i])
        best_assignment[:, i] = 0
        best_assignment[best, i] = 1
        assert np.sum(best_assignment[:, i]) == 1

    best_weights = np.zeros(shape=(num_clusters, num_objectives))
    new_loss = 0
    for i in range(num_clusters):
        w, loss = weight_calculator(best_assignment[i])
        best_weights[i] = w
        new_loss += loss
    return best_assignment, best_weights, min_loss


def solve_exact(mus, sigmas, ids, num_clusters=2, num_objectives=3, verbose=False, optimization_iterations=10,
                girl=False):
    num_agents = len(mus)
    assert num_clusters <= num_agents
    weight_calculator = make_weights_assignment_function(mus, sigmas, ids, num_objectives=num_objectives,
                                                         verbose=verbose,
                                                         num_iters=optimization_iterations,
                                                         girl=girl)
    min_loss = np.inf

    candidate_assignment = np.zeros((num_clusters, num_agents))
    candidate_omega = np.zeros((num_clusters, num_objectives))
    agents = np.array([i for i in range(num_agents)])
    count = 0
    for partition in algorithm_u(agents, num_clusters):
        candidate_loss = 0

        for i, p in enumerate(partition):
            candidate_assignment[i] = 0
            for index in p:
                candidate_assignment[i, index] = 1
            w, loss = weight_calculator(candidate_assignment[i])
            candidate_omega[i] = w
            if loss is None:
                print("None loss")
            candidate_loss += loss

        if candidate_loss < min_loss:
            min_loss = candidate_loss
            best_assignment = np.copy(candidate_assignment)
            best_omega = np.copy(candidate_omega)
        count += 1
        if verbose:
            print("Finished Partition %d" %(count))
            print("Best Assignment")
            print(best_assignment)
            print("Best omega")
            print(best_omega)
            print("Best loss")
            print(min_loss)
    return best_assignment, best_omega, min_loss