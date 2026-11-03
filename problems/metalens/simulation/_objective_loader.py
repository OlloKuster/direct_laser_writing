from problems.metalens.simulation.objective import objective_em_f, objective_em_heat_f


def objective_loader(objective: str, *args):
    """
    Loads the respective objective as a function. Whether the optimization is robust is set by the filter.
    :param objective: Selects which objective function will be returned. The filter function should take the density
                   and additional parameters as an input.
                   Modes:
                    "em_only": Returns the pure electromagnetic optimization function.
                    "em_heat": Returns the (softmax) connectivity optimization function.
                    "robust_em_heat": Same as "em_heat".
    :return: The objective function with rho -> L(rho)
    """
    if objective == "em_only":
        init_projection, filter, projection, currents, resolution, init_value, _, _ = args
        return objective_em_f(init_projection, filter, projection, currents, resolution, init_value)

    if objective in ("em_heat", "robust_em_heat"):
        init_projection, filter, projection, currents, resolution, init_value_em, init_value_mat, init_value_void = args
        return objective_em_heat_f(init_projection, filter, projection, currents, resolution, (init_value_em, init_value_mat, init_value_void))
    return None
