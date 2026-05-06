import jax
import numpy as np

from settings._setting_loader import setting_loader


def main(resolution, betas, setting, loss_hist, em_loss_hist, opt, max_evals, load=None, eval=False, full_bin=False,
         run_id=0):
    jax.config.update("jax_enable_x64", True)

    run = setting["run"]

    return run(resolution, betas, setting, loss_hist, em_loss_hist, opt=opt, max_evals=max_evals, load=load, eval=eval,
               full_bin=full_bin, run_id=run_id)


if __name__ == "__main__":
    setting_init = setting_loader("metalens", "dlw_regular")
    setting_final = setting_loader("metalens", "dlw_regular")
    eval = True
    resolution = 14
    loss_hist = []
    em_loss_hist = []
    betas_init = [16, 32]
    betas_final = [np.inf]

    # loss_hist, em_loss_hist = main(resolution, betas_init, setting_init, loss_hist, em_loss_hist, max_evals=25, opt="nlopt",
    #                                eval=eval, full_bin=False, run_id=0)
    loss_hist, em_loss_hist = main(resolution, betas_final, setting_final, loss_hist, em_loss_hist, max_evals=1, opt="nlopt",
                                   eval=eval, full_bin=False, run_id=0)
