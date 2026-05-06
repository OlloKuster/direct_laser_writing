from dataclasses import dataclass
import tidy3d as td

from problems.power_splitter.simulation.config_structure import ConfigSim


@dataclass
class Sources:
    source = td.ModeSource(
        source_time=td.GaussianPulse(freq0=ConfigSim.freq0, fwidth=ConfigSim.fwidth),
        center=ConfigSim.pos_source,
        size=ConfigSim.size_source,
        mode_index=0,
        mode_spec=td.ModeSpec(num_modes=1),
        direction='+'
    )


@dataclass
class Monitors:
    mode_monitor = td.ModeMonitor(
        center=ConfigSim.pos_monitor,
        size=ConfigSim.size_monitor,
        freqs=ConfigSim.f_eval,
        mode_spec=td.ModeSpec(num_modes=ConfigSim.num_modes),
        name='Mode Monitor'
    )
    flux_monitor = td.FluxMonitor(
        center=ConfigSim.pos_monitor,
        size=ConfigSim.size_monitor,
        freqs=ConfigSim.f_eval,
        name='Flux Monitor'
    )

    field_monitor_source = td.FieldMonitor(
        center=(ConfigSim.pos_source[0] + 0.5, ConfigSim.pos_source[1], ConfigSim.pos_source[2]),
        size=ConfigSim.size_monitor,
        freqs=[ConfigSim.freq0],
        name="Source Field Monitor"
    )

    field_monitor_center = td.FieldMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, td.inf),
        freqs=[ConfigSim.freq0],
        name="Field Monitor"
    )
    eps_monitor = td.PermittivityMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[ConfigSim.freq0],
        name="Permittivity Monitor"
    )

    eps_monitor_full = td.PermittivityMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, td.inf),
        freqs=[ConfigSim.freq0],
        name="Permittivity Monitor Full"
    )
