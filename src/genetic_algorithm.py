import numpy as np
import numba
import os
import yaml
from pathlib import Path
from scipy.spatial.distance import cdist
from time import perf_counter

with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG_YAML = yaml.safe_load(f)
CONFIG = CONFIG_YAML["genetic_algorithm_parameters"]

population_size = CONFIG["population_size"]
max_generations = CONFIG["max_generations"]
selection_size = CONFIG["selection_size"]
order_crossover_rate = CONFIG["order_crossover_rate"]
swap_mutation_rate = CONFIG["swap_mutation_rate"]
min_change = CONFIG["min_change"]
convergence_generation = CONFIG["convergence_generation"]
two_opt_percentile = CONFIG["two_opt_percentile"]

NO_GIL = True
CAN_PARALLEL = True
CAN_CACHE = True


@numba.njit(nogil=NO_GIL, cache=CAN_CACHE)
def create_tour(n):
    tour = np.arange(n)
    np.random.shuffle(tour)
    return tour


@numba.njit(parallel=CAN_PARALLEL, nogil=NO_GIL, cache=CAN_CACHE)
def create_population(population_size, n):
    population = np.empty((population_size, n), dtype=np.int64)

    for i in numba.prange(population_size):
        population[i] = create_tour(n)
    return population


@numba.njit(nogil=NO_GIL, cache=CAN_CACHE)
def evaluate_tour(n, tour, distance_matrix):
    fitness = 0.0

    for i in range(n - 1):
        fitness += distance_matrix[tour[i], tour[i + 1]]
    fitness += distance_matrix[tour[-1], tour[0]]
    return fitness


@numba.njit(parallel=CAN_PARALLEL, nogil=NO_GIL, cache=CAN_CACHE)
def evaluate_population(population_size, n, population, distance_matrix):
    fitnesses = np.empty(population_size, dtype=np.float64)

    for i in numba.prange(population_size):
        fitnesses[i] = evaluate_tour(n, population[i], distance_matrix)
    return fitnesses


@numba.njit(nogil=NO_GIL, cache=CAN_CACHE)
def selection(fitnesses, selection_size):
    elite_index = np.random.randint(len(fitnesses))
    elite_fitness = fitnesses[elite_index]

    for i in range(1, selection_size):
        tour_index = np.random.randint(len(fitnesses))
        if fitnesses[tour_index] < elite_fitness:
            elite_fitness = fitnesses[tour_index]
            elite_index = tour_index
    return elite_index


@numba.njit(nogil=NO_GIL, cache=CAN_CACHE)
def order_crossover(order_crossover_rate, parent_1, parent_2, n):
    if np.random.random() <= order_crossover_rate:
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
def swap_mutation(swap_mutation_rate, tour, n):
    if np.random.random() < swap_mutation_rate:
        mutation = tour.copy()
        gene_1 = np.random.randint(n)
        gene_2 = np.random.randint(n)

        while gene_1 == gene_2:
            gene_2 = np.random.randint(n)

        mutation[gene_1], mutation[gene_2] = mutation[gene_2], mutation[gene_1]
        return mutation
    return tour


@numba.njit(nogil=NO_GIL, cache=CAN_CACHE)
def get_elite(fitnesses, population):
    elite_index = np.argmin(fitnesses)
    elite_tour = population[elite_index].copy()
    elite_fitness = np.float64(fitnesses[elite_index])
    return elite_tour, elite_fitness


@numba.njit(parallel=CAN_PARALLEL, nogil=NO_GIL, cache=CAN_CACHE)
def evolve_population(
    population_size,
    fitnesses,
    selection_size,
    population,
    order_crossover_rate,
    swap_mutation_rate,
    n,
):
    new_population = np.empty((population_size - 1, n), dtype=np.int64)

    for i in numba.prange(1, population_size):
        tour_1 = selection(fitnesses, selection_size)
        tour_2 = selection(fitnesses, selection_size)

        while tour_1 == tour_2:
            tour_1 = selection(fitnesses, selection_size)

        parent_1 = population[tour_1]
        parent_2 = population[tour_2]
        new_population[i - 1] = order_crossover(
            order_crossover_rate,
            parent_1,
            parent_2,
            n,
        )

        new_population[i - 1] = swap_mutation(
            swap_mutation_rate,
            new_population[i - 1],
            n,
        )
    return new_population


@numba.njit(parallel=CAN_PARALLEL, nogil=NO_GIL, cache=CAN_CACHE)
def percentile_two_opt(
    tour_count,
    population,
    fitnesses,
    distance_matrix,
    n,
    threshold
):
    for i in numba.prange(tour_count):
        if fitnesses[i] <= threshold:
            population[i] = two_opt(n, population[i], distance_matrix)
            fitnesses[i] = evaluate_tour(n, population[i], distance_matrix)
    return population, fitnesses


@numba.njit(nogil=NO_GIL, cache=CAN_CACHE)
def two_opt(n, tour, distance_matrix):
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
                    distance_matrix[node_1, node_2] + distance_matrix[node_3, node_4]
                    > distance_matrix[node_1, node_3] + distance_matrix[node_2, node_4]
                ):
                    tour[i : j + 1] = tour[i : j + 1][: : -1]
                    has_improved = True
    return tour


def run_genetic_algorithm(instance):
    print(f"  n: {instance.n}, optimality: {instance.optimality}")
    print("hyperparameters")
    print(f"  population size: {population_size}", end=", ")
    print(f"max generations: {max_generations}")

    print(f"  selection size: {selection_size}", end=", ")
    print(f"order crossover rate: {order_crossover_rate}", end=", ")
    print(f"swap mutation rate: {swap_mutation_rate}")

    print(f"  minimum change: {min_change}", end=", ")
    print(f"max stagnant generations: {convergence_generation}")
    
    print(f"  two-opt percentile: {two_opt_percentile}")

    print()
    print(f"{'gen':<5} {'time':<10} {'elite':<15} {'error':<6}")

    optimality = instance.optimality
    n = instance.n
    nodes = instance.nodes
    distance_matrix = np.round(cdist(nodes, nodes, "euclidean")).astype(np.int64)

    start_time = perf_counter()
    generation_start_time = perf_counter()

    population = create_population(population_size, n)
    fitnesses = evaluate_population(population_size, n, population, distance_matrix)
    elite_tour, elite_fitness = get_elite(fitnesses, population)

    if optimality is not None:
        error = (elite_fitness - optimality) / optimality * 100
    else:
        error = None

    generation_time = perf_counter() - generation_start_time

    stagnant_generations = 0
    for generation in range(1, max_generations + 1):
        generation_start_time = perf_counter()

        new_population = evolve_population(
            population_size,
            fitnesses,
            selection_size,
            population,
            order_crossover_rate,
            swap_mutation_rate,
            n,
        )

        new_fitnesses = evaluate_population(
            population_size - 1,
            n,
            new_population,
            distance_matrix,
        )

        threshold = np.percentile(new_fitnesses, two_opt_percentile)
        new_population, new_fitnesses = percentile_two_opt(
            population_size - 1,
            new_population,
            new_fitnesses,
            distance_matrix,
            n,
            threshold,
        )

        elite_population = np.empty((population_size, n), dtype=np.int64)
        elite_population[0] = elite_tour.copy()
        elite_population[1:] = new_population

        elite_fitnesses = np.empty(population_size, dtype=np.float64)
        elite_fitnesses[0] = elite_fitness
        elite_fitnesses[1:] = new_fitnesses

        population, fitnesses = elite_population, elite_fitnesses
        elite_tour_draw, elite_fitness_draw = get_elite(fitnesses, population)

        if elite_fitness_draw < elite_fitness - min_change:
            elite_tour = elite_tour_draw.copy()
            elite_fit = elite_fitness_draw
            stagnant_generations = 0
        else:
            stagnant_generations += 1
            
        elite_tour = two_opt(n, elite_tour.copy(), distance_matrix)
        elite_fitness = evaluate_tour(n, elite_tour, distance_matrix)

        if optimality is not None:
            error = (elite_fitness - optimality) / optimality * 100
        else: 
            error = None

        generation_time = perf_counter() - generation_start_time

        print(f"{generation:>5} {generation_time:<10.3f}", end=" ")
        print(f"{elite_fitness:<15.3f} {error:<6.3f}")

        if (
            stagnant_generations >= convergence_generation
            or generation >= max_generations
            or (optimality is not None and elite_fitness == optimality)
        ):
            end_time = perf_counter()
            return elite_fitness
    return elite_fitness