import numpy as np
import matplotlib.pyplot as plt
import tidy3d.web as web
import scipy

from problems.mode_converter.simulation.config_structure import ConfigSimMode
from problems.mode_converter.simulation.simulation import make_sim_tidy


def upload_sim(seed):
    if seed != None:
        np.random.seed(seed)
    rho_0 = np.random.rand(ConfigSimMode.nx, ConfigSimMode.ny, ConfigSimMode.nz)

    sim = make_sim_tidy(rho_0)
    # sim.plot(z=-ConfigSimMode.lz / 2 + ConfigSimMode.thickness_substrate + ConfigSimMode.wg_height / 2 )
    sim.plot(y=0)
    plt.show()
    # job = web.Job(simulation=sim, task_name="test")
    # estimated_cost = web.estimate_cost(job.task_id)

    # sim_data = web.run(sim, task_name="name", folder_name="mode_converter", verbose=True)
    # e_x = sim_data["Mode Field Monitor"].Ex.values.squeeze()
    # print(e_x.shape)
    # plt.imshow(np.abs(e_x).T, origin='lower')
    # plt.show()



    # print(estimated_cost)



if __name__ == "__main__":
    upload_sim(123)