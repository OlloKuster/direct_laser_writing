from filtering.dose_model.config_print import ConfigPrint
import numpy as np

def setting_loader(system: str, setup: str):
    if system == "metalens":
        if setup == "dlw_regular":
            setting_dict = {
                "objectives": "em_heat",
                "filters": "dose_conv",
                "filter_factor": 1,
                "plotter_eval": "eval_regular",
                "plotter_final": "final_regular",
                "projection": "ssp_jax",
                "projection_values": ConfigPrint.rho_th_GT,
                "init_projection": "tanh_jax",
                "init_projection_values": 0.5,
                "optimizers": "torch_jax",
                "conversions": "torch",
                "backconversions": "torch2np",

                "init_em": "em_only",
                "init_heat": "heat_only"

            }
            return setting_dict

        if setup == "dlw_robust":
            setting_dict = {
                "objectives": "robust_em_heat",
                "filters": "dose_conv",
                "filter_factor": 1,
                "plotter_eval": "eval_robust",
                "plotter_final": "final_robust",
                "projection": "robust_ssp_jax",
                "projection_values": [0.9 * ConfigPrint.rho_th_GT, ConfigPrint.rho_th_GT, 1.1 * ConfigPrint.rho_th_GT],
                "init_projection": "tanh_jax",
                "init_projection_values": 0.5,
                "optimizers": "torch_jax",
                "conversions": "torch",
                "backconversions": "torch2np",

                "init_em": "em_only",
                "init_heat": "heat_only"

            }
            return setting_dict

        if setup == "normal_gauss":
            setting_dict = {
                "objectives": "em_heat",
                "filters": "gauss_jax",
                "filter_factor": np.sqrt(3) / 2,
                "plotter_eval": "eval_regular",
                "plotter_final": "final_regular",
                "projection": "ssp_jax",
                "projection_values": 0.5,
                "init_projection": "tanh_jax",
                "init_projection_values": 0.5,
                "optimizers": "jax",
                "conversions": "None",
                "backconversions": "None",

                "init_em": "em_only",
                "init_heat": "heat_only"

            }
            return setting_dict

        else:
            raise Exception("Setup not found")
    else:
        raise Exception("System not found")