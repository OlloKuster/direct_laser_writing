from dataclasses import dataclass
import tidy3d as td

from problems.polarization_splitter.simulation.config_structure import ConfigSim


@dataclass
class Sources:
    source_te = td.ModeSource(
        source_time=td.GaussianPulse(freq0=ConfigSim.freq0, fwidth=ConfigSim.fwidth),
        center=ConfigSim.pos_source,
        size=ConfigSim.size_source,
        mode_index=1,
        mode_spec=td.ModeSpec(num_modes=ConfigSim.num_modes),
        direction='+'
    )
    source_tm = td.ModeSource(
        source_time=td.GaussianPulse(freq0=ConfigSim.freq0, fwidth=ConfigSim.fwidth),
        center=ConfigSim.pos_source,
        size=ConfigSim.size_source,
        mode_index=0,
        mode_spec=td.ModeSpec(num_modes=ConfigSim.num_modes),
        direction='+'
    )


@dataclass
class Monitors:
    mode_monitor_te = td.ModeMonitor(
        center=ConfigSim.pos_monitor_te,
        size=ConfigSim.size_monitor_te,
        freqs=[ConfigSim.freq0],
        mode_spec=td.ModeSpec(num_modes=ConfigSim.num_modes),
        name='Mode Monitor Horizontal'
    )
    mode_monitor_tm = td.ModeMonitor(
        center=ConfigSim.pos_monitor_tm,
        size=ConfigSim.size_monitor_tm,
        freqs=[ConfigSim.freq0],
        mode_spec=td.ModeSpec(num_modes=ConfigSim.num_modes),
        name='Mode Monitor Vertical'
    )

    field_monitor_source = td.FieldMonitor(
        center=(ConfigSim.pos_source[0] + 0.5, ConfigSim.pos_source[1], ConfigSim.pos_source[2]),
        size=ConfigSim.size_monitor,
        freqs=[ConfigSim.freq0],
        name="Source Field Monitor"
    )

    field_monitor_center = td.FieldMonitor(
        center=(0, 0, -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.wg_height / 2),
        size=(td.inf, td.inf, 0),
        freqs=[ConfigSim.freq0],
        name="Field Monitor"
    )

    eps_monitor = td.PermittivityMonitor(
        center=(0, 0, -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.wg_height / 2),
        size=(td.inf, td.inf, 0),
        freqs=[ConfigSim.freq0],
        name="Permittivity Monitor"
    )
