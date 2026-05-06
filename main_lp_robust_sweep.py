import jax
import numpy as np

from dispenser import Dispenser
from settings._setting_loader import setting_loader


def main(resolution, betas, setting, loss_hist, em_loss_hist, opt, max_evals, load=None, eval=False, full_bin=False,
         run_id=0):
    """
    This function can be run to reproduce the results of the paper regarding the robustness of the designs.
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

    run = Dispenser.LENS3D
    return run(resolution, betas, setting, loss_hist, em_loss_hist, opt=opt, max_evals=max_evals, load=load, eval=eval, full_bin=full_bin, run_id=run_id)


if __name__ == "__main__":
    setting = setting_loader("metalens", "dlw_robust")
    eval = True
    device_id = 0
    resolution = 14
    loss_hist = []
    em_loss_hist = []
    betas = [np.inf]
    run_id = 1

    lps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    for lp in lps:
        loss_hist = []
        em_loss_hist = []
        setting["lp_deviation"] = lp
        loss_hist, em_loss_hist = main(resolution, betas, setting, loss_hist, em_loss_hist, max_evals=20,
                                       opt="nlopt", eval=eval, full_bin=False, run_id=run_id,
                                       load=f'/scratch/local/okuster/Code/00_Main_Projects/dlw_params/problems/metalens/plots/data_base.h5')
        run_id = run_id + 1
