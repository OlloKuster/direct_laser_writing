from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class ConfigG:
    PATH = "/scratch/local/okuster/data/dlw_metalens/plots_feature_size_sweep/"