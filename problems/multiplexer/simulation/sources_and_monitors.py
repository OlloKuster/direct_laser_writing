from dataclasses import dataclass
import tidy3d as td

from problems.multiplexer.simulation.config_structure import ConfigSim


@dataclass
class Sources:
    source = td.ModeSource(
        source_time=td.GaussianPulse(freq0=ConfigSim.freq0, fwidth=ConfigSim.fwidth),
        center=ConfigSim.pos_source,
        size=ConfigSim.size_source,
        mode_index=0,
        mode_spec=td.ModeSpec(num_modes=ConfigSim.num_modes),
        direction='+'
    )


@dataclass
class Monitors:
    mode_monitor = td.ModeMonitor(
        center=ConfigSim.pos_monitor,
        size=ConfigSim.size_monitor,
        freqs=ConfigSim.eval_freqs,
        mode_spec=td.ModeSpec(num_modes=ConfigSim.num_modes),
        name=f'Mode Monitor {ConfigSim.wavelength}'
    )

    field_monitor_source = td.FieldMonitor(
        center=(ConfigSim.pos_source[0]+0.5,  ConfigSim.pos_source[1], ConfigSim.pos_source[2]),
        size=(td.inf, td.inf, 0),
        freqs=ConfigSim.eval_freqs,
        name="Source Field Monitor"
    )

    field_monitor_center = td.FieldMonitor(
        center=(0, 0, -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.wg_height / 2),
        size=(td.inf, td.inf, 0),
        freqs=ConfigSim.eval_freqs,
        name="Field Monitor"
    )

    eps_monitor = td.PermittivityMonitor(
        center=(0, 0, -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.wg_height / 2),
        size=(td.inf, td.inf, 0),
        freqs=[ConfigSim.freq0],
        name="PermittivityMonitor"
    )
