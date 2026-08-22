import numpy as np
import numba
import os
from pathlib import Path
from scipy.spatial.distance import cdist
from time import perf_counter

NO_GIL = True
CAN_PARALLEL = True
CAN_CACHE = True

pop_size = 200
max_gens = 1000
selection_size = 7
ox_rate = 0.85
sm_rate = 0.03
min_change = 1e-3
convergence_gen = 50
two_opt_pt = 10


@numba.njit(nogil=NO_GIL, cache=CAN_CACHE)
def rand_tour(n):
    tour = np.arange(n)
    np.random.shuffle(tour)
    return tour


@numba.njit(parallel=CAN_PARALLEL, nogil=NO_GIL, cache=CAN_CACHE)
def init_pop(pop_size, n):
    pop = np.empty((pop_size, n), dtype=np.int64)
    for i in numba.prange(pop_size):
        pop[i] = rand_tour(n)
    return pop


@numba.njit(nogil=NO_GIL, cache=CAN_CACHE)
def calc_fit(n, tour, dist_matrix):
    fit = 0.0
    for i in range(n - 1):
        fit += dist_matrix[tour[i], tour[i + 1]]
    fit += dist_matrix[tour[-1], tour[0]]
    return fit


@numba.njit(parallel=CAN_PARALLEL, nogil=NO_GIL, cache=CAN_CACHE)
def calc_pop_fits(pop_size, n, pop, dist_matrix):
    fits = np.empty(pop_size, dtype=np.float64)
    for i in numba.prange(pop_size):
        fits[i] = calc_fit(n, pop[i], dist_matrix)
    return fits


@numba.njit(nogil=NO_GIL, cache=CAN_CACHE)
def selection(fits, selection_size):
    elite_index = np.random.randint(len(fits))
    elite_fit = fits[elite_index]
    for i in range(1, selection_size):
        tour_index = np.random.randint(len(fits))
        if fits[tour_index] < elite_fit:
            elite_fit = fits[tour_index]
            elite_index = tour_index
    return elite_index


@numba.njit(nogil=NO_GIL, cache=CAN_CACHE)
def order_crossover(ox_rate, parent_1, parent_2, n):
    if np.random.random() <= ox_rate:
        lower = np.random.randint(n)
        upper = np.random.randint(n)

        while lower == upper:
            upper = np.random.randint(n)

        if lower > upper:
            lower, upper = upper, lower

        child = np.full(n, -1, dtype=np.int64)
        child[lower:upper] = parent_1[lower:upper]

        fill = []
        subset = set(parent_1[lower:upper])
        for i in parent_2:
            if i not in subset:
                fill.append(i)

        j = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[j]
                j += 1
        return child
    else:
        if np.random.random() < 0.5:
            return parent_2.copy()
        return parent_1.copy()


@numba.njit(nogil=NO_GIL, cache=CAN_CACHE)
def swap_mutation(sm_rate, tour, n):
    if np.random.random() < sm_rate:
        mutation = tour.copy()
        gene_1 = np.random.randint(n)
        gene_2 = np.random.randint(n)

        while gene_1 == gene_2:
            gene_2 = np.random.randint(n)

        mutation[gene_1], mutation[gene_2] = mutation[gene_2], mutation[gene_1]
        return mutation
    return tour


@numba.njit(nogil=NO_GIL, cache=CAN_CACHE)
def get_elite(fits, pop):
    elite_index = np.argmin(fits)
    elite_tour = pop[elite_index].copy()
    elite_fit = np.float64(fits[elite_index])
    return elite_tour, elite_fit


@numba.njit(parallel=CAN_PARALLEL, nogil=NO_GIL, cache=CAN_CACHE)
def evolve_pop(pop_size, fits, selection_size, pop, ox_rate, sm_rate, n):
    new_pop = np.empty((pop_size - 1, n), dtype=np.int64)
    for i in numba.prange(1, pop_size):
        tour_1 = selection(fits, selection_size)
        tour_2 = selection(fits, selection_size)

        while tour_1 == tour_2:
            tour_1 = selection(fits, selection_size)

        parent_1 = pop[tour_1]
        parent_2 = pop[tour_2]
        new_pop[i - 1] = order_crossover(ox_rate, parent_1, parent_2, n)
        new_pop[i - 1] = swap_mutation(sm_rate, new_pop[i - 1], n)
    return new_pop


@numba.njit(parallel=CAN_PARALLEL, nogil=NO_GIL, cache=CAN_CACHE)
def two_opt_percentile(num_of_tours, pop, fits, dist_matrix, n, threshold):
    for i in numba.prange(num_of_tours):
        if fits[i] <= threshold:
            pop[i] = two_opt(n, pop[i], dist_matrix)
            fits[i] = calc_fit(n, pop[i], dist_matrix)
    return pop, fits


@numba.njit(nogil=NO_GIL, cache=CAN_CACHE)
def two_opt(n, tour, dist_matrix):
    has_improved = True
    while has_improved:
        has_improved = False

        for i in range(1, n - 2):
            for j in range(i + 1, n):
                node_1 = tour[i - 1]
                node_2 = tour[i]
                node_3 = tour[j]
                node_4 = tour[(j + 1) % n]

                if (
                    dist_matrix[node_1, node_2] + dist_matrix[node_3, node_4]
                    > dist_matrix[node_1, node_3] + dist_matrix[node_2, node_4]
                ):
                    tour[i : j + 1] = tour[i : j + 1][: : -1]
                    has_improved = True
    return tour


def run_ga(instance):
    opt_fit = instance.opt_fit
    n = instance.get_n()
    nodes = instance.get_nodes()
    dist_matrix = np.round(cdist(nodes, nodes, "euclidean")).astype(np.int64)
    start_time = perf_counter()
    gen_start_time = perf_counter()

    pop = init_pop(pop_size, n)
    fits = calc_pop_fits(pop_size, n, pop, dist_matrix)
    elite_tour, elite_fit = get_elite(fits, pop)

    if opt_fit is not None:
        error = (elite_fit - opt_fit) / opt_fit * 100
    else:
        error = None

    gen_time = perf_counter() - gen_start_time

    stagnant_gens = 0
    for gen in range(1, max_gens + 1):
        gen_start_time = perf_counter()

        new_pop = evolve_pop(pop_size, fits, selection_size, pop, ox_rate, sm_rate, n)
        new_fits = calc_pop_fits(pop_size - 1, n, new_pop, dist_matrix)
        threshold = np.percentile(new_fits, two_opt_pt)
        new_pop, new_fits = two_opt_percentile(
            pop_size - 1,
            new_pop,
            new_fits,
            dist_matrix,
            n,
            threshold,
        )

        elite_pop = np.empty((pop_size, n), dtype=np.int64)
        elite_pop[0] = elite_tour.copy()
        elite_pop[1:] = new_pop

        elite_fits = np.empty(pop_size, dtype=np.float64)
        elite_fits[0] = elite_fit
        elite_fits[1:] = new_fits

        pop, fits = elite_pop, elite_fits
        candidate_elite_tour, candidate_elite_fit = get_elite(fits, pop)
        if candidate_elite_fit < elite_fit - min_change:
            elite_tour = candidate_elite_tour.copy()
            elite_fit = candidate_elite_fit
            stagnant_gens = 0
        else:
            stagnant_gens += 1
            
        elite_tour = two_opt(n, elite_tour.copy(), dist_matrix)
        elite_fit = calc_fit(n, elite_tour, dist_matrix)

        if opt_fit is not None:
            error = (elite_fit - opt_fit) / opt_fit * 100
        else: 
            error = None

        gen_time = perf_counter() - gen_start_time

        if (
            stagnant_gens >= convergence_gen
            or gen >= max_gens
            or (opt_fit is not None and elite_fit == opt_fit)
        ):
            end_time = perf_counter()
            return elite_fit