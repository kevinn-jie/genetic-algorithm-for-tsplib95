from genetic_algorithm import genetic_algorithm
from load_instance import Instance

if __name__ == "__main__":
	instance = Instance("berlin52")
	result = genetic_algorithm(instance)
	print(result)