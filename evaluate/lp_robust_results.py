import h5py
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv


font = {'family': 'sans-serif',
        'size': 16}

matplotlib.rc('font', **font)

path_feature_size = "/scratch/local/okuster/Code/00_Main_Projects/dlw_params/evaluate/data/lp_robust_sweep/"

lp_diff = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
inds_lp_diff = np.linspace(0, len(lp_diff)-1, len(lp_diff))
lp_diff_rel = np.array(lp_diff) * 100
# i = 0

em_loss_eroded = np.zeros_like(lp_diff_rel)
em_loss_regular = np.zeros_like(lp_diff_rel)
em_loss_dilated = np.zeros_like(lp_diff_rel)
loss_list_eroded = np.zeros_like(lp_diff_rel)
loss_list_normal = np.zeros_like(lp_diff_rel)
loss_list_dilated = np.zeros_like(lp_diff_rel)
loss_final = np.zeros_like(lp_diff_rel)


for i in inds_lp_diff:
    with h5py.File(path_feature_size + f"data_{int(i)}_inf.h5") as f:
        grp = f["lens_3d"]
        em_loss = grp["em_loss"][:]
        loss = grp["loss"][:]
        print(grp.keys())
        eps_dilation = grp['eps_dilation'][:]
        eps_erosion = grp['eps_erosion'][:]
        eps_normal = grp['eps_normal'][:]
        em_loss_eroded[int(i)] = em_loss[-1][0]
        em_loss_regular[int(i)] = em_loss[-1][1]
        em_loss_dilated[int(i)] = em_loss[-1][2]
        loss_list_eroded[int(i)] = em_loss[-1][0]
        loss_list_normal[int(i)] = em_loss[-1][1]
        loss_list_dilated[int(i)] = em_loss[-1][2]
        loss_final[int(i)] = loss[-1]
        #
        # data_eroded = pv.wrap(eps_erosion)
        # data_normal = pv.wrap(eps_normal)
        # data_dilated = pv.wrap(eps_dilation)
        #
        # p = pv.Plotter(off_screen=True)
        # p.add_mesh(data_eroded.contour(), cmap='binary')
        # p.camera_position = 'yz'
        # p.camera.elevation = 20
        # p.camera.azimuth = - 45
        # p.remove_scalar_bar()
        # p.camera.zoom(1.3)
        # p.show(screenshot=f'plots/robust_structures/{i}_eroded.png')
        # p.close()
        #
        # p = pv.Plotter(off_screen=True)
        # p.add_mesh(data_normal.contour(), cmap='binary')
        # p.camera_position = 'yz'
        # p.camera.elevation = 20
        # p.camera.azimuth = - 45
        # p.remove_scalar_bar()
        # p.camera.zoom(1.3)
        # p.show(screenshot=f'plots/robust_structures/{i}_normal.png')
        # p.close()
        #
        # p = pv.Plotter(off_screen=True)
        # p.add_mesh(data_dilated.contour(), cmap='binary')
        # p.camera_position = 'yz'
        # p.camera.elevation = 20
        # p.camera.azimuth = - 45
        # p.remove_scalar_bar()
        # p.camera.zoom(1.3)
        # p.show(screenshot=f'plots/robust_structures/{i}_dilated.png')
        # p.close()

fig, axs = plt.subplots(1, 1, figsize=(6, 6))
# plt.plot(em_loss_eroded)
# plt.plot(em_loss_regular)
# plt.plot(em_loss_dilated)
axs.plot(lp_diff_rel, loss_list_eroded, color='grey', label='Eroded', linestyle=':')
axs.plot(lp_diff_rel, loss_list_normal, '-ro', color='black', label='Normal')
axs.plot(lp_diff_rel, loss_list_dilated, color='grey', label='Dilated', linestyle='--')
# axs.plot(lp_diff_rel, loss_final, color='black', label='Dilated', linestyle='--')
axs.set_ylim(0, 7)
axs.set_xlabel(r"Laser Power Deviation$\,$(%)", fontsize=24)
axs.set_ylabel(r"$\mathcal{L}_\text{EM}$", fontsize=24)
plt.legend()
plt.savefig("plots/dlw.png")
plt.close()





