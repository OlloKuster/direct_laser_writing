import jax
import torch

from dispenser import Dispenser

def main():
    jax.config.update("jax_enable_x64", True)
    torch.cuda.empty_cache()
    run = Dispenser.LENS3D
    run(40, [16, 32, jax.numpy.inf])


if __name__ == "__main__":
    main()