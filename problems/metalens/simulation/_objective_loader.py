from problems.metalens.simulation.objective import objective_em_f, objective_em_heat_f, objective_heat_f


def objective_loader(objective: str, *args):
    if objective == "em_only":
        currents, resolution, init_value = args
        return objective_em_f(currents, resolution, init_value)

    if objective == "heat_only":
        return objective_heat_f()

    if objective == "em_heat":
        currents, resolution, init_value_em, init_value_mat, init_value_void = args
        return objective_em_heat_f(currents, resolution, (init_value_em, init_value_mat, init_value_void))

