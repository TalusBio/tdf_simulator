from __future__ import annotations

import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from tdf_simulator.config import RunConfig, TDFConfig
from tdf_simulator.frames import FrameData, FrameInfoBuilder
from tdf_simulator.tdf_sql import TDFInfoBuilder


class PeakSimulator(ABC):
    """Abstract class that defines the interface for a peak simulator.

    The idea is to have a stateful object that can generate peak data
    provided a time and a window information.
    """

    @abstractmethod
    def generate_frame_data(
        self,
        time: float,
        window_info: list[WindowInfo] | None,
    ) -> tuple[FrameData, int]:
        """Generate the frame data at a given time."""
        raise NotImplementedError


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


@dataclass
class TDFSimulator:
    """Class that generates TDF data."""

    folder_name: os.PathLike
    tdf_config: TDFConfig
    run_config: RunConfig
    peak_simulator = None

    def generate_data(self) -> None:
        """Generate the TDF data.

        This function is the main loop that will generate the TDF data.
        And should really be the only public function of this class.
        """
        self.report_start()
        info_builder = FrameInfoBuilder(self.tdf_config, self.run_config)
        frames_template = info_builder.build_frames_df_template()
        self.report_frame_expects(frames_template)
        windows = info_builder.build_dia_frame_msms_windows()
        frames_info = info_builder.build_frame_msms_info(
            frames_id=frames_template["Id"],
            frames_msms_type=frames_template["MsMsType"],
        )
        grouped_windows = {}
        for g, sdg in windows.groupby("WindowGroup"):
            grouped_windows[g] = [
                WindowInfo.from_dict(x) for x in sdg.to_dict(orient="records")
            ]

        frame_to_window = {}
        for _i, row in frames_info.iterrows():
            frame_to_window[row["Frame"]] = row["WindowGroup"]

        curr_frame_offset = 0
        frame_offsets = []
        max_intensities = []
        summed_intensities = []
        num_peaks = []

        folder = Path(self.folder_name)
        if not folder.exists():
            logger.info("Creating folder: {}", folder)
            folder.mkdir(parents=True)

        bin_out = folder / "analysis.tdf_bin"
        logger.info("Generating binary file: {}", bin_out)
        with open(bin_out, "wb") as f:
            for _i, row in tqdm(frames_template.iterrows()):
                row_dict = row.to_dict()
                frame_offsets.append(curr_frame_offset)
                if row_dict["MsMsType"] == 0:
                    frame_data, curr_frame_peaks = self._frame_data_from_row_dict(
                        row_dict, window_info=None
                    )
                else:
                    frame_data, curr_frame_peaks = self._frame_data_from_row_dict(
                        row_dict,
                        window_info=grouped_windows[frame_to_window[row_dict["Id"]]],
                    )
                max_intensities.append(frame_data.max_intensity)
                summed_intensities.append(frame_data.summed_intensities)
                num_peaks.append(curr_frame_peaks)
                frame_bin, curr_frame_offset = frame_data.pack_data(curr_frame_offset)
                f.write(frame_bin)

        max_peaks = max(num_peaks)
        tdf_info_builder = TDFInfoBuilder(self.tdf_config, self.run_config)

        frames = info_builder.complete_frames_df(
            frames_template,
            frame_offsets,
            max_intensities,
            summed_intensities,
            num_peaks,
        )

        self._write_tdf_file(
            global_meta_data=tdf_info_builder.build_global_metadata(max_peaks),
            frames=frames,
            dia_frame_msms_info=frames_info,
            dia_frame_msms_window_groups=info_builder.build_frame_msms_window_grops(),
            dia_frame_msms_windows=windows,
        )

    def report_start(self) -> None:
        """Report the start of the simulation."""
        logger.info("Generating TDF data")
        logger.info("TDF Config: {}", self.tdf_config)
        logger.info("Run Config: {}", self.run_config)

    def report_frame_expects(self, frames_template: pd.DataFrame) -> None:
        """Report the frame template that will be used to generate the data."""
        logger.info("Frames template:")
        logger.info("\n" + repr(frames_template))

    def _frame_data_from_row_dict(
        self, row_dict: dict[str, Any], window_info: list[WindowInfo] | None
    ) -> tuple[FrameData, int]:
        """Internal function that converts a row dictionary to a FrameData object.

        Each row comes from the frames template and contains the information
        to generate a frame.

        It can either pass the information to the peak simulator or generate
        a dummy frame if no peak simulator is provided.
        """
        # {'Id': 1, 'Time': 0.1, 'Polarity': '+', 'ScanMode': 9,
        #  'MsMsType': 0, 'NumScans': 709, 'AccumulationTime': 100,
        #  'RampTime': 100}
        if self.peak_simulator:
            frame_data, curr_frame_peaks = self.peak_simulator.generate_frame_data(
                time=row_dict["Time"],
                window_info=window_info,
            )
        else:
            frame_data, curr_frame_peaks = FrameData.generate_dummy(
                num_scans=row_dict["NumScans"], offset=0
            )
        return frame_data, curr_frame_peaks

    def _write_tdf_file(
        self,
        global_meta_data: pd.DataFrame,
        frames: pd.DataFrame,
        dia_frame_msms_info: pd.DataFrame,
        dia_frame_msms_window_groups: pd.DataFrame,
        dia_frame_msms_windows: pd.DataFrame,
    ) -> None:
        """Write the TDF file to disk.

        This function is meant to be called after all the binary data is generated.
        And the data frames that compose the sql database are ready to be written.
        """
        # TODO implement schema checking...
        tdf_file_name = Path(self.folder_name) / "analysis.tdf"
        logger.info(f"Writing TDF file to: {tdf_file_name}")
        with sqlite3.connect(tdf_file_name) as sql_database_connection:
            global_meta_data.to_sql(
                "GlobalMetaData", sql_database_connection, index=False
            )
            frames.to_sql("Frames", sql_database_connection, index=False)
            dia_frame_msms_info.to_sql(
                "DiaFrameMsMsInfo", sql_database_connection, index=False
            )

            dia_frame_msms_window_groups.to_sql(
                "DiaFrameMsMsWindowGroups", sql_database_connection, index=False
            )

            dia_frame_msms_windows.to_sql(
                "DiaFrameMsMsWindows", sql_database_connection, index=False
            )
