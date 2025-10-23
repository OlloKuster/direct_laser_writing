import jax
import numpy as np
import torch
import nlopt
import time

from filtering.dose_model.config_print import ConfigPrint
from optimizer import config


def optimiser(rho, objective, filter, projection, mode):
    """
    The optimiser function
    :return:
    """
    loss_hist = []

    def select_f(mode):
        if mode == "torch_jax":
            return f_torch_jax
        if mode == "jax":
            return f_jax

    class fom_em_torch_f(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            grad_em_sim_f = jax.value_and_grad(objective)
            value_em_sim, grad_em_sim = grad_em_sim_f(projection(x.detach().cpu().numpy().astype(np.float64)))
            ctx.save_for_backward(torch.tensor(np.array(grad_em_sim), device='cuda', requires_grad=True))
            return torch.tensor(np.array(value_em_sim), device='cuda', requires_grad=True)

        @staticmethod
        def backward(ctx, grad):
            grad_em_sim, = ctx.saved_tensors
            return grad_em_sim

    def f_torch_jax(x, g):
        start = time.time()
        x = np.reshape(x, rho.shape)
        rho_0 = torch.tensor(x, device='cuda', requires_grad=True)
        rho_final = filter(rho_0)
        fom = fom_em_torch_f.apply(rho_final)
        fom.backward(retain_graph=True)
        grad_torch = rho_0.grad
        value = fom.detach().cpu().numpy()
        grad = grad_torch.detach().cpu().numpy()
        value = float(value)  # Requires np float and not jax.numpy float
        print(f"value: {value}")
        print(f"grad: {np.sum(grad)}")
        print(f"iteration: {config.cur_it}")
        config.cur_it += 1
        loss_hist.append(value)
        if g.size > 0:
            g[:] = grad.ravel()
        end = time.time()
        print(f"time: {end - start}")
        if eval:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, sharex=True)
            rho_ = rho_0.detach().cpu().numpy()
            rho_ = np.concatenate((rho_, np.flip(rho_, axis=0)), axis=0)
            rho_ = np.concatenate((rho_, np.flip(rho_, axis=1)), axis=1)
            # rho_ = np.concatenate((rho_, np.flip(rho_, axis=2)), axis=2)
            ax[0].imshow(rho_[rho_.shape[0]//2].T, origin='lower', cmap='binary', vmin=0, vmax=1)
            # ax[0].set_xlabel(r"x ($\mathrm{\mu}$m)", fontsize=12)
            ax[0].set_ylabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
            rho_f = rho_final.detach().cpu().numpy()
            rho_f = np.concatenate((rho_f, np.flip(rho_f, axis=0)), axis=0)
            rho_f = np.concatenate((rho_f, np.flip(rho_f, axis=1)), axis=1)
            # rho_f = np.concatenate((rho_f, np.flip(rho_f, axis=2)), axis=2)
            rho_f = projection(rho_f)
            ax[1].imshow(rho_f[rho_f.shape[0]//2].T, origin='lower', cmap='binary', vmin=0, vmax=1)
            ax[1].set_xlabel(r"x ($\mathrm{\mu}$m)", fontsize=12)
            ax[1].set_ylabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
            plt.savefig(f"problems/metalens/plots/progression/rho_{config.cur_it:03d}.png")
            plt.close()
            if config.cur_it % config.MAXEVAL == 0:
                config.ind += 1
        return value

    def f_jax(x, g):
        start = time.time()
        rho_0 = np.reshape(x, rho.shape)
        rho_final = filter(rho_0)
        value, grad = jax.value_and_grad(objective)(rho_final)
        value = float(value)  # Requires np float and not jax.numpy float
        print(f"value: {value}")
        print(f"grad: {np.sum(grad)}")
        print(f"iteration: {config.cur_it}")
        config.cur_it += 1
        loss_hist.append(value)
        if g.size > 0:
            g[:] = grad.ravel()
        end = time.time()
        print(f"time: {end - start}")
        if eval:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, sharex=True)
            rho_ = rho_0
            rho_ = np.concatenate((rho_, np.flip(rho_, axis=0)), axis=0)
            rho_ = np.concatenate((rho_, np.flip(rho_, axis=1)), axis=1)
            # rho_ = np.concatenate((rho_, np.flip(rho_, axis=2)), axis=2)
            ax[0].imshow(rho_[rho_.shape[0]//2].T, origin='lower', cmap='binary', vmin=0, vmax=1)
            # ax[0].set_xlabel(r"x ($\mathrm{\mu}$m)", fontsize=12)
            ax[0].set_ylabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
            rho_f = rho_final
            rho_f = np.concatenate((rho_f, np.flip(rho_f, axis=0)), axis=0)
            rho_f = np.concatenate((rho_f, np.flip(rho_f, axis=1)), axis=1)
            # rho_f = np.concatenate((rho_f, np.flip(rho_f, axis=2)), axis=2)
            rho_f = projection(rho_f)
            ax[1].imshow(rho_f[rho_f.shape[0]//2].T, origin='lower', cmap='binary', vmin=0, vmax=1)
            ax[1].set_xlabel(r"x ($\mathrm{\mu}$m)", fontsize=12)
            ax[1].set_ylabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
            plt.savefig(f"problems/metalens/plots/progression/rho_{config.cur_it:03d}.png")
            plt.close()
            if config.cur_it % config.MAXEVAL == 0:
                config.ind += 1
        return value

    f = select_f(mode)

    opt = nlopt.opt(config.OPTIMISER, rho.size)
    # opt.set_param('tolg', 1e-12)
    opt.set_min_objective(f)
    opt.set_maxeval(config.MAXEVAL)
    # opt.set_ftol_abs(config.FTOL_ABS)
    # opt.set_ftol_rel(config.FTOL_REL)
    # opt.set_stopval(1e-4)
    opt.set_upper_bounds(config.UPPER_BOUNDS)
    opt.set_lower_bounds(config.LOWER_BOUNDS)

    rho_opt = opt.optimize(rho.ravel())
    rho_opt = rho_opt.reshape(rho.shape)

    return rho_opt, loss_hist

