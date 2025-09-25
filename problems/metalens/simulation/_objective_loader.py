from problems.metalens.simulation.objective import objective_em_f


def objective_loader(objective: str, *args):
    if objective == "em_only":
        projection, currents, resolution, init_value = args
        return objective_em_f(projection, currents, resolution, init_value)
