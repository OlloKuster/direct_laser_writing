import jax
from settings._setting_loader import setting_loader


def main(resolution, betas, setting, loss_hist, em_loss_hist, opt, max_evals, load=None, eval=False, full_bin=False,
         run_id=0):
    """
    This function can be run to check the resolution dependencies of the results.
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
    setting = setting_loader("metalens", "gauss_em_only")
    eval = False
    resolution = 10
    betas = [32]
    run_id = 0

    resolutions = [16]
    for resolution in resolutions:
        loss_hist = []
        em_loss_hist = []
        loss_hist, em_loss_hist = main(resolution, betas, setting, loss_hist, em_loss_hist, max_evals=20, opt="nlopt", eval=eval, full_bin=False, run_id=7)

        run_id = run_id + 1
