def load_optimality_file(file):
    optimality_file = {}

    if not file.exists():
        return optimality_file

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith("#"):
                continue

            if ':' not in line:
                continue

            key, value = line.split(":", 1)

            try:
                optimality_file[key.strip()] = int(value.strip())
            except ValueError:
                continue
    return optimality_file