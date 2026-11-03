from problems.metalens.simulation.objective import objective_em_f, objective_robust_em_heat_f, objective_heat_f, \
    objective_em_heat_f


def objective_loader(objective: str, *args):
    """
    Loads the respective objective as a function.
    :param objective: Selects which objective function will be returned. The filter function should take the density
                   and additional parameters as an input.
                   Modes:
                    "em_only": Returns the pure electromagnetic optimization function.
                    "heat_only": Returns the pure heat_eval optimization function.
                    "em_heat": Returns the (softmax) connectivity optimization function.
                    "conic_jax": Returns a cone filter which uses Jax.
                    "robust_em_heat": Returns the (softmax) connectivity optimization function for a dlw
                                      optimization.
    :return: The objective function with rho -> L(rho)
    """
    if objective == "em_only":
        init_projection, filter, projection, currents, resolution, init_value, _, _ = args
        return objective_em_f(init_projection, filter, projection, currents, resolution, init_value)

    if objective == "heat_only":
        init_projection, filter, projection = args
        return objective_heat_f(init_projection, filter, projection)

    if objective == "em_heat":
        init_projection, filter, projection, currents, resolution, init_value_em, init_value_mat, init_value_void = args
        return objective_em_heat_f(init_projection, filter, projection, currents, resolution, (init_value_em, init_value_mat, init_value_void))

    if objective == "robust_em_heat":
        init_projection, filter, projection, currents, resolution, init_value_em, init_value_mat, init_value_void = args
        return objective_robust_em_heat_f(init_projection, filter, projection, currents, resolution, (init_value_em, init_value_mat, init_value_void))
    return None

