from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import tomli
import tomli_w
from loguru import logger


@dataclass
class TDFConfig:
    SCAN_MODE: int = 9
    MZ_MIN: float = 100.000000
    MZ_MAX: float = 1600.000000
    NUM_TOF_BINS: int = 628_556  # 434_064 # 638_437 # 638_437
    IM_MIN: float = 0.5
    IM_MAX: float = 1.5
    NUM_SCANS: int = 709
    ACCUMULATION_TIME: int = 100
    RAMP_TIME: int = 100

    @property
    def bottleneck_time_ms(self) -> int:
        return max(self.ACCUMULATION_TIME, self.RAMP_TIME)

    @classmethod
    def from_tdf_connection(cls, conn: sqlite3.Connection) -> TDFConfig:
        dat = conn.execute("SELECT Key, Value FROM GlobalMetadata;")
        dat = {v[0]: v[1] for v in dat}
        return cls(
            SCAN_MODE=9,
            MZ_MIN=300.0,  # TODO .... find how to infer this one...
            MZ_MAX=1600.0,  # TODO .... find how to infer this one...
            NUM_TOF_BINS=int(dat["DigitizerNumSamples"]),
            IM_MIN=float(dat["OneOverK0AcqRangeLower"]),
            IM_MAX=float(dat["OneOverK0AcqRangeUpper"]),
            NUM_SCANS=int(
                max(x[0] for x in conn.execute("SELECT NumScans FROM Frames;"))
            ),
            ACCUMULATION_TIME=int(
                max(x[0] for x in conn.execute("SELECT AccumulationTime FROM Frames;"))
            ),
            RAMP_TIME=int(
                max(x[0] for x in conn.execute("SELECT RampTime FROM Frames;"))
            ),
        )

    @classmethod
    def from_toml_file(cls, file_path: str) -> TDFConfig:
        with open(file_path, "rb") as f:
            config = tomli.load(f)
        return cls(**config["tdf_config"])

    def to_toml(self, file_path: str, *, append: bool = False) -> None:
        mode = "ab" if append else "wb"
        with open(file_path, mode) as f:
            tomli_w.dump({"tdf_config": asdict(self)}, f)


@dataclass
class RunConfig:
    num_cycles: int
    frames_per_cycle: int
    num_dia_window_groups: int
    scan_groups_per_window_group: int

    @classmethod
    def from_tdf_connection(cls, conn: sqlite3.Connection) -> RunConfig:
        df = pd.read_sql("SELECT * FROM Frames;", conn)
        window_df = pd.read_sql("SELECT * FROM DiaFrameMsMsWindows;", conn)
        max_rt_minutes = df["Time"].max() / 60
        return cls.from_window_df(
            window_df, max_rt_minutes, TDFConfig.from_tdf_connection(conn)
        )

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
            num_cycles=(
                max_rt_seconds // (time_per_frame_seconds * frames_per_cycle)
            ).item(),
            frames_per_cycle=frames_per_cycle.item(),
            num_dia_window_groups=num_dia_window_groups.item(),
            scan_groups_per_window_group=scan_groups_per_window_group,
        )

    @classmethod
    def from_toml_file(cls, file_path: str) -> RunConfig:
        with open(file_path, "rb") as f:
            config = tomli.load(f)
        return cls(**config["run_config"])

    def to_toml(self, file_path: str, *, append: bool = False) -> None:
        mode = "ab" if append else "wb"
        with open(file_path, mode) as f:
            tomli_w.dump({"run_config": asdict(self)}, f)


@dataclass
class WindowInfo:
    """Class that holds the information about a DIA window."""

    # {'WindowGroup': 2, 'ScanNumBegin': 30,
    #  'ScanNumEnd': 180, 'IsolationMz': 800,
    #  'IsolationWidth': 50, 'CollisionEnergy': 42}
    window_group: int
    scan_num_begin: int
    scan_num_end: int
    isolation_mz: float
    isolation_width: float
    collision_energy: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WindowInfo:
        """Convert a dictionary to a WindowInfo object.

        Args:
            d (dict[str, Any]): The dictionary to convert.
                It assumes the keys will have the capitalization
                of the parent table in sql. WhichIsCamelCase.
        """
        return cls(
            window_group=d["WindowGroup"],
            scan_num_begin=d["ScanNumBegin"],
            scan_num_end=d["ScanNumEnd"],
            isolation_mz=d["IsolationMz"],
            isolation_width=d["IsolationWidth"],
            collision_energy=d["CollisionEnergy"],
        )

    @classmethod
    def from_toml_file(cls, file_path: str) -> list[WindowInfo]:
        with open(file_path, "rb") as f:
            config = tomli.load(f)
        return [cls(**x) for x in config["window_info"]]

    @classmethod
    def to_toml(
        cls, file_path: str, window_info: list[WindowInfo], *, append: bool = False
    ) -> None:
        mode = "ab" if append else "wb"
        with open(file_path, mode) as f:
            tomli_w.dump({"window_info": [asdict(x) for x in window_info]}, f)

    @classmethod
    def from_tdf_connection(cls, conn: sqlite3.Connection) -> list[WindowInfo]:
        df = pd.read_sql("SELECT * FROM DiaFrameMsMsWindows;", conn)
        return [cls.from_dict(x) for x in df.to_dict(orient="records")]


if __name__ == "__main__":
    import argparse

    from tdf_simulator.converters import SpecConverter, Tof2MzConverter

    parser = argparse.ArgumentParser(description="Prints the TDF and Run Configs.")
    parser.add_argument("tdf_file", type=str, help="The path to the TDF file.")
    parser.add_argument(
        "--toml",
        type=str,
        help="The path to the toml file to save the configs.",
        default=None,
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.tdf_file)
    tdf_config = TDFConfig.from_tdf_connection(conn)
    run_config = RunConfig.from_tdf_connection(conn)
    windows = WindowInfo.from_tdf_connection(conn)
    spec_converter = SpecConverter.from_connection(conn)
    tof2mz_converter = Tof2MzConverter.from_connection(conn)
    conn.close()
    logger.info(tdf_config)
    logger.info(run_config)

    if args.toml:
        tdf_config.to_toml(args.toml)
        run_config.to_toml(args.toml, append=True)
        WindowInfo.to_toml(args.toml, windows, append=True)

        # Check that they can be read
        tdf_config = TDFConfig.from_toml_file(args.toml)
        run_config = RunConfig.from_toml_file(args.toml)
        windows = WindowInfo.from_toml_file(args.toml)
        logger.info(tdf_config)
        logger.info(run_config)
        logger.info(windows)
