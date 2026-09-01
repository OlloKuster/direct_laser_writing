import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch
import jax
import scipy

from filtering.dose_model._dose_filter import dose_filter_f
from filtering.dose_model.config_print import ConfigPrint
from problems.metalens.simulation.config_structure import ConfigSim
from projection.SSP.subpixel_smoothed_projection import f2bin_smooth, ssp_proj_jax_f
from projection.tanh.tanh_projection import tanh_filter_jax_f

jax.config.update("jax_enable_x64", True)


resolution = 10
beta = 10

init_proj = tanh_filter_jax_f(beta=beta)
dlw_filter = dose_filter_f(resolution)
ssp_filter = ssp_proj_jax_f(ConfigPrint.rho_th_GT, beta, resolution)

def objective_fd(x):
    return np.sum(x)

def objective_false(x):
    return jnp.sum(x)

def objective_correct(x):
    return jnp.sum(ssp_filter(x))




class FomEmTorchF_false(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        grad_obj_function = jax.value_and_grad(objective_false)
        val, grad_ad = grad_obj_function(ssp_filter(x.detach().cpu().numpy().astype(np.float64)))
        ctx.save_for_backward(torch.tensor(np.array(grad_ad), device='cuda', requires_grad=True))
        return torch.tensor(np.array(val), device='cuda', requires_grad=True)

    @staticmethod
    def backward(ctx, grad):
        grad_em_sim, = ctx.saved_tensors
        return grad_em_sim * grad

class FomEmTorchF_correct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        grad_obj_function = jax.value_and_grad(objective_correct)
        val, grad_ad = grad_obj_function(x.detach().cpu().numpy().astype(np.float64))
        ctx.save_for_backward(torch.tensor(np.array(grad_ad), device='cuda', requires_grad=True))
        return torch.tensor(np.array(val), device='cuda', requires_grad=True)

    @staticmethod
    def backward(ctx, grad):
        grad_em_sim, = ctx.saved_tensors
        return grad_em_sim * grad




rho_0 = np.random.rand(int(np.ceil((ConfigSim.rho_shape[0] + ConfigSim.buffer_side) * resolution)),
                     int(np.ceil((ConfigSim.rho_shape[1] + ConfigSim.buffer_side) * resolution)),
                     int(np.ceil(ConfigSim.rho_shape[2] * resolution)))

rho_0 = scipy.ndimage.gaussian_filter(rho_0, sigma=1)


writing_pattern = torch.tensor(np.array(init_proj(rho_0)), device='cuda', requires_grad=True)
accumulated_dose = dlw_filter(writing_pattern)
printed_design = ssp_filter(accumulated_dose.detach().cpu().numpy())

fig, ax = plt.subplots(1, 3)
ax[0].imshow(writing_pattern.detach().cpu().numpy()[writing_pattern.shape[0]//2].T, origin='lower')
ax[1].imshow(accumulated_dose.detach().cpu().numpy()[accumulated_dose.shape[0]//2].T, origin='lower')
ax[2].imshow(printed_design[printed_design.shape[0]//2].T, origin='lower')
plt.show()


print(f"FD FoM:\t{objective_fd(printed_design)}")
print(f"AD FoM correct:\t{FomEmTorchF_correct.apply(accumulated_dose)}")
print(f"AD FoM false:\t{FomEmTorchF_false.apply(accumulated_dose)}")


ad_fom_correct = FomEmTorchF_correct.apply(accumulated_dose)
ad_fom_correct.backward(retain_graph=True)
ad_grad_correct = writing_pattern.grad
ad_grad_correct = ad_grad_correct.detach().cpu().numpy()


print(f"grad correct mean:\t{np.mean(ad_grad_correct)}")
fig, ax = plt.subplots(1, 2)
ax[0].imshow(ad_grad_correct[ad_grad_correct.shape[0]//2] / np.max(ad_grad_correct) .T, origin='lower')

ad_fom_false = FomEmTorchF_false.apply(accumulated_dose)
ad_fom_false.backward(retain_graph=True)
ad_grad_false = writing_pattern.grad
ad_grad_false = ad_grad_false.detach().cpu().numpy()

print(f"grad false mean:\t{np.mean(ad_grad_false)}")
ax[1].imshow(ad_grad_false[ad_grad_false.shape[0]//2] / np.max(ad_grad_false) .T, origin='lower')
plt.show()

deltas = [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

delta_location = np.random.randint((0, 0, 0), (rho_0.shape[0], rho_0.shape[1], rho_0.shape[2]), size=3)

delta_location = [rho_0.shape[0]//2, rho_0.shape[1]//2, rho_0.shape[2]//2]
fd_grads = []

diffs_correct = []
diffs_false = []

for delta in deltas:

    writing_pattern_delta = writing_pattern.clone()
    writing_pattern_delta[delta_location[0], delta_location[1], delta_location[2]] += delta
    accumulated_dose_delta = dlw_filter(writing_pattern_delta).detach().cpu().numpy()
    printed_design_delta = ssp_filter(accumulated_dose_delta)

    fd_grad = (objective_fd(printed_design_delta) - objective_fd(printed_design)) / delta
    fd_grads.append(fd_grad)

    diff_correct = np.abs(ad_grad_correct[delta_location[0], delta_location[1], delta_location[2]] - fd_grad) / fd_grad

    diff_false = np.abs(ad_grad_false[delta_location[0], delta_location[1], delta_location[2]] - fd_grad) / fd_grad

    print(diff_correct)

    diffs_correct.append(diff_correct)

    diffs_false.append(diff_false)

plt.plot(deltas, diffs_correct)
#plt.plot(deltas, diffs_false)
plt.xscale('log')
plt.yscale('log')
plt.show()

