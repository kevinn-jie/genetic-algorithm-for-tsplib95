# Genetic Algorithm for TSPLIB95

Last updated: 8/27/2026

## Description

This repository was dedicated to my 2025-2027 IB Extended Essay research. Subsequently, this project also acted as a practice project; this was not made to be a usable tool. This project is a Python, numba-accelerated genetic algorithm implementation for 2D Euclidean TSPLIB95 instances.

### Genetic Algorithm Implementation

1. Selection
2. Order crossover
3. Swap mutation
4. Two-opt top percentile
5. Elitism
6. Convergence check

### Genetic Algorithm Hyperparameters

* Population size
* Maximum generations
* Selection size
* Order crossover rate
* Swap mutation rate
* Minimum change
* Convergence generation
* Two-opt percentile

## Stack

### Built With

* Python (3.12.10)
    * TSPLIB95 (0.7.1)
    * NumPy (2.5.2)
    * SciPy (1.18.1)
    * Numba (0.67.0)
    * PyYAML (6.0.3)

## Installation

1. Virtual environment and activation

    ```bash
    python3.12 -m venv .venv
    .venv/Scripts/activate
    ```

2. Install requirements

    ```bash
    pip install -r requirements.txt
    ```

3. TSPLIB95 data in `data/`, but can be found at `pdrozdowski/TSPLib.Net` repository: [https://github.com/pdrozdowski/TSPLib.Net/tree/master/TSPLIB95/tsp](https://github.com/pdrozdowski/TSPLib.Net/tree/master/TSPLIB95/tsp).

## Usage

1. If TSP data not in `data/`, confirm `self.OPTIMALITY_FILE` and `self.instance_FILE` paths in `load_instance.py`

    ```python
    ...

    class Instance:
        def __init__(self, name):
            self.ROOT = Path(__file__).resolve().parents[1]
            self.OPTIMALITY_FILE = self.ROOT / "data" / "bestSolutions.txt" # confirm path
            self.name = name
            self.optimality = load_optimality_file(self.OPTIMALITY_FILE).get(self.name)
            self.instance_file = self.ROOT / "data" / f"{name}.tsp" # confirm path
            
    ...
    ```

2. Adjust genetic algorithm hyperparameters in `config.yaml`

    ```yaml
    genetic_algorithm_parameters: 
      population_size: 200
      max_generations: 1000
      selection_size: 7
      order_crossover_rate: 0.85
      swap_mutation_rate: 0.03
      min_change: 0.001
      convergence_generation: 50
      two_opt_percentile: 100
    ```

3. Adjust `main.py` using `genetic_algorithm.py` and `load_instance.py`

    ```python
    from genetic_algorithm import run_genetic_algorithm
    from load_instance import Instance

    if __name__ == "__main__":
        instance = Instance("fl417") # change to instance of choice
        run_genetic_algorithm(instance) # run genetic algorithm on instance
    ```

## Project Status

The project is finalized; data has been collected. This repository will no longer receive future updates.