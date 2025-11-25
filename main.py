import jax
import torch
import numpy as np

from dispenser import Dispenser


def main(resolution, betas, eval):
    jax.config.update("jax_enable_x64", True)
    torch.cuda.empty_cache()
    run = Dispenser.LENS3D
    run(resolution, betas, eval=eval)


if __name__ == "__main__":
    main(8, [1, np.inf, np.inf], eval=True)
