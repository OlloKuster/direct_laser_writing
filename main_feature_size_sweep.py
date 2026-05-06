import jax
import numpy as np

from settings._setting_loader import setting_loader


def main(resolution, betas, setting, loss_hist, em_loss_hist, opt, max_evals, load=None, eval=False, full_bin=False,
         run_id=0):
    """
    This function can be run to reproduce the results of the paper regarding the minimum feature size.
    :param resolution: Resolution of the problem (in px/lenght)
    :param betas: Which binarization levels are used.
    :param setting: Which setting is used for optimization.
    :param loss_hist: List to track the Figure of Merit.
    :param em_loss_hist: List to track the EM Figure of Merit.
    :param opt: Which optimizer is used. Options are "nlopt" or "optax".
    :param max_evals: How many evaluations/iterations are done in the optimization.
    :param load: If set, will load the density from the specified file.
    :param eval: If intermediate plots will be plotted.
    :param full_bin: If the designs are fully binarized.
    :param run_id: Tracking id for the runs.
    :return: Loss and EM loss history.
    """
    jax.config.update("jax_enable_x64", True)

    run = setting["run"]

    return run(resolution, betas, setting, loss_hist, em_loss_hist, opt=opt, max_evals=max_evals, load=load, eval=eval, full_bin=full_bin, run_id=run_id)



if __name__ == "__main__":
    setting_init = setting_loader("metalens", "no_filter")
    setting_final = setting_loader("metalens", "gauss_em_only")
    eval = True
    resolution = 14
    betas_init = [16, 32]
    betas_final = [np.inf]
    run_id = 0

    feature_size_factor = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for fact in feature_size_factor:
        loss_hist = []
        em_loss_hist = []
        setting_init["objectives"] = "em_only"
        setting_final["filter_factor"] = fact / np.sqrt(3)
        loss_hist, em_loss_hist = main(resolution, betas_final, setting_final, loss_hist, em_loss_hist, max_evals=15,
                                       opt="nlopt",
                                       eval=eval, full_bin=False, run_id=run_id,
                                       load=f'/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/metalens/plots/data_base.h5')

        run_id = run_id + 1
