from genetic_algorithm import run_genetic_algorithm
from load_instance import Instance

if __name__ == "__main__":
	instance = Instance("fl417")
	run_genetic_algorithm(instance)