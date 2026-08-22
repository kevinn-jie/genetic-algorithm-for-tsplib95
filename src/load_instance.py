import numpy as np
import tsplib95
from pathlib import Path


def get_opt_fits(file):
    opt_fits = {}

    if not file.exists():
        return opt_fits

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith("#"):
                continue

            if ':' not in line:
                continue

            key, value = line.split(":", 1)

            try:
                opt_fits[key.strip()] = int(value.strip())
            except ValueError:
                continue
    return opt_fits


class Instance:
    def __init__(self, name):
        self.ROOT_DIR = Path(__file__).resolve().parents[1]
        self.OPT_FITS_FILE = self.ROOT_DIR / "data" / "tsplib95" / "opt_fits.txt"

        self.name = name
        self.file = self.ROOT_DIR / "data" / "tsplib95" / "instances" / f"{name}.tsp"
        self.instance = tsplib95.load(self.file)
        self.type = self.instance.edge_weight_type
        nodes = self.instance.node_coords.values()
        self.nodes = np.array(list(nodes))
        self.opt_fit = get_opt_fits(self.OPT_FITS_FILE).get(self.name)
        self.n = self.nodes.shape[0]

    def get_nodes(self):
        return self.nodes

    def get_n(self):
        return self.n