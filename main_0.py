import jax
import numpy as np

from settings._setting_loader import setting_loader


def main(resolution, betas, setting, loss_hist, em_loss_hist, opt, max_evals, load=None, eval=False, run_id=0):
    """
    Main function, starts the optimization process with the specified settings. The way it is set up,
    multiple runs can be done sequentially with different settings.
    :param resolution: Resolution of the problem (in px/lenght)
    :param betas: Which binarization levels are used.
    :param setting: Which setting is used for optimization.
    :param loss_hist: List to track the Figure of Merit.
    :param em_loss_hist: List to track the EM Figure of Merit.
    :param opt: Which optimizer is used. Options are "nlopt" or "optax".
    :param max_evals: How many evaluations/iterations are done in the optimization.
    :param load: If set, will load the density from the specified file.
    :param eval: If intermediate plots will be plotted.
    :param run_id: Tracking id for the runs.
    :return: Loss and EM loss history.
    """
    jax.config.update("jax_enable_x64", True)

    run = setting["run"]

    return run(resolution, betas, setting, loss_hist, em_loss_hist, opt=opt, max_evals=max_evals, load=load, eval=eval,
               run_id=run_id)


if __name__ == "__main__":
    setting_init = setting_loader("metalens", "no_filter")
    setting_final = setting_loader("metalens", "dlw_regular")
    eval = True
    resolution = 14
    loss_hist = []
    em_loss_hist = []
    betas_init = [16, 32]
    betas_final = [np.inf]

    loss_hist, em_loss_hist = main(resolution, betas_init, setting_init, loss_hist, em_loss_hist, max_evals=15, opt="nlopt",
                                   eval=eval, run_id=1)
    loss_hist, em_loss_hist = main(resolution, betas_init, setting_init, loss_hist, em_loss_hist, max_evals=15, opt="nlopt",
                                   eval=eval, run_id=2)
    loss_hist, em_loss_hist = main(resolution, betas_init, setting_init, loss_hist, em_loss_hist, max_evals=15, opt="nlopt",
                                   eval=eval, run_id=3)
    loss_hist, em_loss_hist = main(resolution, betas_init, setting_init, loss_hist, em_loss_hist, max_evals=15, opt="nlopt",
                                   eval=eval, run_id=4)
    loss_hist, em_loss_hist = main(resolution, betas_init, setting_init, loss_hist, em_loss_hist, max_evals=15, opt="nlopt",
                                   eval=eval, run_id=5)
    # loss_hist, em_loss_hist = main(resolution, betas_final, setting_final, loss_hist, em_loss_hist, max_evals=1, opt="nlopt",
    #                                eval=eval, run_id=0)
