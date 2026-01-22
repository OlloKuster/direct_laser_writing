from dataclasses import dataclass
import tidy3d as td

from problems.mode_converter.simulation.config_structure import ConfigSimMode


@dataclass
class Sources:
    source = td.ModeSource(
        source_time=td.GaussianPulse(freq0=ConfigSimMode.freq0, fwidth=ConfigSimMode.fwidth),
        center=ConfigSimMode.pos_source,
        size=ConfigSimMode.size_source,
        mode_index=0,
        mode_spec=td.ModeSpec(num_modes=ConfigSimMode.num_modes),
        direction='+'
    )


@dataclass
class Monitors:
    mode_monitor = td.ModeMonitor(
        center=ConfigSimMode.pos_monitor,
        size=ConfigSimMode.size_monitor,
        freqs=[ConfigSimMode.freq0],
        mode_spec=td.ModeSpec(num_modes=ConfigSimMode.num_modes),
        name='Mode Monitor'
    )

    field_monitor_source = td.FieldMonitor(
        center=(ConfigSimMode.pos_source[0]+0.5,  ConfigSimMode.pos_source[1], ConfigSimMode.pos_source[2]),
        size=(td.inf, td.inf, 0),
        freqs=[ConfigSimMode.freq0],
        name="Source Field Monitor"
    )

    field_monitor_mode = td.FieldMonitor(
        center=ConfigSimMode.pos_monitor,
        size=ConfigSimMode.size_monitor,
        freqs=[ConfigSimMode.freq0],
        name="Mode Field Monitor"
    )

    field_monitor_center = td.FieldMonitor(
        center=(0, 0, -ConfigSimMode.lz / 2 + ConfigSimMode.thickness_substrate + ConfigSimMode.rho_size[2] / 2),
        size=(td.inf, td.inf, 0),
        freqs=[ConfigSimMode.freq0],
        name="Field Monitor"
    )

    eps_monitor = td.PermittivityMonitor(
        center=(0, 0, -ConfigSimMode.lz / 2 + ConfigSimMode.thickness_substrate + ConfigSimMode.rho_size[2] / 2),
        size=(td.inf, td.inf, 0),
        freqs=[ConfigSimMode.freq0],
        name="PermittivityMonitor"
    )
