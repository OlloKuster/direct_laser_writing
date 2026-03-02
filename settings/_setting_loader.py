from dispenser import Dispenser
from filtering.dose_model.config_print import ConfigPrint
import numpy as np


def setting_loader(system: str, setup: str):
    """
    Setup the settings we want to optimize here. The examples from the paper can be loaded in from here directly.
    :param system: What is the optical system we want to optimize.
    :param setup: Which particular problem is being optimized.
    :return: Dictionary of the relevant simulation parameters.
    """
    if system == "metalens":
        if setup == "dlw_regular":
            setting_dict = {
                "run": Dispenser.LENS3D,
                "objectives": "em_heat",  # Objective function(s) of the problem.
                "filters": "dose_conv",  # Filter function for the optimization.
                "filter_factor": 1,  # Factor for the size of the filter (1 is 1um).
                "plotter_eval": "eval_regular",  # Which plotting function is used for the evaluation.
                "plotter_final": "final_regular",  # Which plotting function is used for the final plotting.
                "projection": "ssp_jax",  # Projection used for the optimization.
                "projection_values": ConfigPrint.rho_th_GT,  # Threshold value used in projection.
                "init_projection": "ssp_jax",  # Initial projection used for the "precompensated" structure.
                "init_projection_values": 0.5,  # Threshold value for the inital projection.
                "optimizers": "torch_jax",  # Which mode the opimizer runs in.
                "conversions": "torch",  # Conversion of the variables while they are being reset in between steps.
                "backconversions": "torch2np",  # Backconversion of the variables while they are being reset in
                #  between steps.

                "target_material": 0.8,
                "target_void": 0.8,

                "init_em": "em_only",  # Initial EM-objective function.
                "init_heat": "heat_only"  # Initial heat_eval-objective function.

            }
            return setting_dict

        if setup == "dlw_robust":
            setting_dict = {
                "run": Dispenser.LENS3D,
                "objectives": "robust_em_heat",
                "filters": "dose_conv",
                "filter_factor": 1,
                "plotter_eval": "eval_robust",
                "plotter_final": "final_robust",
                "projection": "robust_ssp_jax",
                "projection_values": [0.4, 0.5, 0.6],
                "init_projection": "ssp_jax",
                "init_projection_values": 0.5,
                "optimizers": "torch_jax",
                "conversions": "torch",
                "backconversions": "torch2np",
                
                "target_material": 0.8,
                "target_void": 0.8,

                "init_em": "em_only",
                "init_heat": "heat_only"

            }
            return setting_dict

        if setup == "normal_gauss":
            setting_dict = {
                "run": Dispenser.LENS3D,
                "objectives": "em_heat",
                "filters": "gauss_jax",
                "filter_factor": 1 / (2*np.sqrt(3)),
                "plotter_eval": "eval_regular",
                "plotter_final": "final_regular",
                "projection": "ssp_jax",
                "projection_values": 0.5,
                "init_projection": "None",
                "init_projection_values": 0.5,
                "optimizers": "jax",
                "conversions": "None",
                "backconversions": "None",
                
                "target_material": 0.8,
                "target_void": 0.8,

                "init_em": "em_only",
                "init_heat": "heat_only"

            }
            return setting_dict
        if setup == "gauss_em_only":
            setting_dict = {
                "run": Dispenser.LENS3D,
                "objectives": "em_only",
                "filters": "gauss_jax",
                "filter_factor": 1 / (4*np.sqrt(3)),
                "plotter_eval": "eval_regular",
                "plotter_final": "final_regular",
                "projection": "ssp_jax",
                "projection_values": ConfigPrint.rho_th_GT,
                "init_projection": "None",
                "init_projection_values": 0.5,
                "optimizers": "jax",
                "conversions": "None",
                "backconversions": "None",
                
                "target_material": 1.3,
                "target_void": 1.3,

                "init_em": "em_only",
                "init_heat": "heat_only"

            }
            return setting_dict

        else:
            raise Exception("Setup not found")

    if system == "mode_converter":
        if setup == "dlw_regular":
            setting_dict = {
                "run": Dispenser.MODECONVERTER,
                "objectives": "em_heat",  # Objective function(s) of the problem.
                "filters": "dose_conv",  # Filter function for the optimization.
                "filter_factor": 1,  # Factor for the size of the filter (1 is 1um).
                "plotter_eval": "mc_eval_regular",  # Which plotting function is used for the evaluation.
                "plotter_final": "mc_final_regular",  # Which plotting function is used for the final plotting.
                "projection": "ssp_jax",  # Projection used for the optimization.
                "projection_values": ConfigPrint.rho_th_GT,  # Threshold value used in projection.
                "init_projection": "tanh_jax",  # Initial projection used for the "precompensated" structure.
                "init_projection_values": ConfigPrint.rho_th_GT,  # Threshold value for the inital projection.
                "optimizers": "torch_jax",  # Which mode the opimizer runs in.
                "conversions": "torch",  # Conversion of the variables while they are being reset in between steps.
                "backconversions": "torch2np",  # Backconversion of the variables while they are being reset in
                #  between steps.

                "init_em": "em_only",  # Initial EM-objective function.
                "init_heat": "heat_only"  # Initial heat_eval-objective function.

            }

            return setting_dict

        if setup == "normal_gauss":
            setting_dict = {
                "objectives": "em_heat",
                "filters": "gauss_jax",
                "filter_factor": 1 / (3 * np.sqrt(3)),
                "plotter_eval": "eval_regular",
                "plotter_final": "final_regular",
                "projection": "ssp_jax",
                "projection_values": ConfigPrint.rho_th_GT,
                "init_projection": "tanh_jax",
                "init_projection_values": 0.5,
                "optimizers": "jax",
                "conversions": "None",
                "backconversions": "None",

                "init_em": "em_only",
                "init_heat": "heat_only"

            }

            return setting_dict

        if setup == "dlw_robust":
            setting_dict = {
                "objectives": "robust_em_heat",
                "filters": "dose_conv",
                "filter_factor": 1,
                "plotter_eval": "mc_eval_robust",
                "plotter_final": "mc_final_robust",
                "projection": "robust_ssp_jax",
                "projection_values": [0.9 * ConfigPrint.rho_th_GT, ConfigPrint.rho_th_GT, 1.1 * ConfigPrint.rho_th_GT],
                "init_projection": "tanh_jax",
                "init_projection_values": 0.5,
                "optimizers": "torch_jax",
                "conversions": "torch",
                "backconversions": "torch2np",

                "target_material": 1.3,
                "target_void": 1.3,

                "init_em": "em_only",
                "init_heat": "heat_only"

            }
            return setting_dict

        else:
            raise Exception("Setup not found")

    if system == "multiplexer":
        if setup == "dlw_regular":
            setting_dict = {
                "objectives": "em_heat",  # Objective function(s) of the problem.
                "filters": "dose_conv",  # Filter function for the optimization.
                "filter_factor": 1,  # Factor for the size of the filter (1 is 1um).
                "plotter_eval": "muliplex_eval_regular",  # Which plotting function is used for the evaluation.
                "plotter_final": "muliplex_final_regular",  # Which plotting function is used for the final plotting.
                "projection": "ssp_jax",  # Projection used for the optimization.
                "projection_values": ConfigPrint.rho_th_GT,  # Threshold value used in projection.
                "init_projection": "ssp_jax",  # Initial projection used for the "precompensated" structure.
                "init_projection_values": ConfigPrint.rho_th_GT,  # Threshold value for the inital projection.
                "optimizers": "torch_jax",  # Which mode the opimizer runs in.
                "conversions": "torch",  # Conversion of the variables while they are being reset in between steps.
                "backconversions": "torch2np",  # Backconversion of the variables while they are being reset in
                #  between steps.

                "init_em": "em_only",  # Initial EM-objective function.
                "init_heat": "heat_only"  # Initial heat_eval-objective function.

            }

            return setting_dict

        else:
            raise Exception("Setup not found")

    else:
        raise Exception("System not found")
