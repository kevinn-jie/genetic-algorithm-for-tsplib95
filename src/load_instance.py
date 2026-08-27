import numpy as np
import tsplib95
from pathlib import Path
from load_optimality_file import load_optimality_file


class Instance:
    def __init__(self, name):
        self.ROOT = Path(__file__).resolve().parents[1]
        self.OPTIMALITY_FILE = self.ROOT / "data" / "tsplib95" / "bestSolutions.txt"
        self.name = name
        self.optimality = load_optimality_file(self.OPTIMALITY_FILE).get(self.name)
        self.instance_file = self.ROOT / "data" / "tsplib95" / f"{name}.tsp"

        print(f"{self.name}")
        print(f"  file: \"{self.instance_file}\"")

        self.instance = tsplib95.load(self.instance_file)
        self.type = self.instance.edge_weight_type
        self.nodes = np.array(list(self.instance.node_coords.values()))
        self.n = self.nodes.shape[0]