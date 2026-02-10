from problems.multiplexer.simulation.objective import objective_heat_f, objective_em_heat_f


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
                    "robust_em_heat": Returns the (softmax) connectivity optimization function for a robust
                                      optimization.
    :return: The objective function with rho -> L(rho)
    """

    if objective == "heat_only":
        return objective_heat_f()

    if objective == "em_heat":
        init_value_mat, init_value_void = args
        return objective_em_heat_f((init_value_mat, init_value_void))

