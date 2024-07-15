from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TDFConfig:
    SCAN_MODE: int = 9
    MZ_MIN: float = 100.000000
    MZ_MAX: float = 1000.000000
    NUM_TOF_BINS: int = 628_556  # 434_064 # 638_437 # 638_437
    IM_MIN: float = 0.5
    IM_MAX: float = 1.5
    NUM_SCANS: int = 709
    ACCUMULATION_TIME: int = 100
    RAMP_TIME: int = 100

    @property
    def bottleneck_time_ms(self) -> int:
        return max(self.ACCUMULATION_TIME, self.RAMP_TIME)


@dataclass
class RunConfig:
    num_cycles: int
    frames_per_cycle: int
    num_dia_window_groups: int
    scan_groups_per_window_group: int

    @classmethod
    def from_window_df(
        cls, df: pd.DataFrame, max_rt_minutes: float, tdf_config: TDFConfig
    ) -> RunConfig:
        max_rt_seconds = max_rt_minutes * 60
        time_per_frame_seconds = tdf_config.bottleneck_time_ms / 1000

        # I am assuming here that all window groups are acquired between
        # every ms1. Thus num_dia_window_groups + 1 == frames_per_cycle
        num_dia_window_groups = df["WindowGroup"].max()
        frames_per_cycle = num_dia_window_groups + 1
        scan_groups_per_window_group = [len(x) for _i, x in df.groupby("WindowGroup")]
        if len(set(scan_groups_per_window_group)) > 1:
            msg = "The number of scan groups per window group is not consistent."
            msg += f" Found: {scan_groups_per_window_group} (expected: only one value)"
            raise ValueError(msg)

        scan_groups_per_window_group = scan_groups_per_window_group[0]

        return RunConfig(
            num_cycles=(max_rt_seconds // (time_per_frame_seconds * frames_per_cycle)),
            frames_per_cycle=frames_per_cycle,
            num_dia_window_groups=num_dia_window_groups,
            scan_groups_per_window_group=scan_groups_per_window_group,
        )
