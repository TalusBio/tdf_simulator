from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyzstd
from loguru import logger

from tdf_simulator.config import RunConfig, TDFConfig
from tdf_simulator.converters import SpecConverter, Tof2MzConverter
from tdf_simulator.utils import is_sorted_asc


@dataclass
class FrameData:
    frame_counts: np.array
    frame_tofs: list[np.array]
    frame_ints: list[np.array]

    def __post_init__(self):
        logger.trace(self.repr_glimpse())
        if sum(self.frame_counts) != sum(len(x) for x in self.frame_ints):
            msg = "Frame counts must sum to the total number of intensities."
            msg += f" Counts: {sum(self.frame_counts)}, "
            msg += f"Intensities: {sum(len(x) for x in self.frame_ints)}"
            raise ValueError(msg)

    def repr_glimpse(self) -> str:
        return "\n".join(
            [
                "\nFrameData(",
                self.repr_glimpse_counts(indent_level=1),
                self.repr_glimpse_tofs(indent_level=1),
                self.repr_glimpse_ints(indent_level=1),
                ")",
            ]
        )

    def __repr_glimpse_nested_arr(
        self, arr_lst: list[np.array], name: str, indent_level
    ) -> str:
        lines = [f"{name}=("]
        lines += [f"\tarray({x[:5]} ... len({len(x)})," for x in arr_lst[:5]]
        lines += ["\t..."]
        lines += [f"\t... len({len(arr_lst)})", ")"]

        lines = [(indent_level * "\t") + line for line in lines]
        return "\n".join(lines)

    def repr_glimpse_counts(self, indent_level) -> str:
        return f"{'\t' * indent_level}frame_counts={self.frame_counts[:5]} ... len({len(self.frame_counts)}),"

    def repr_glimpse_tofs(self, indent_level) -> str:
        return self.__repr_glimpse_nested_arr(
            self.frame_tofs, "frame_tofs", indent_level=indent_level
        )

    def repr_glimpse_ints(self, indent_level) -> str:
        return self.__repr_glimpse_nested_arr(
            self.frame_ints, "frame_ints", indent_level=indent_level
        )

    @property
    def max_intensity(self) -> np.array:
        return max(np.max(ints) for ints in self.frame_ints)

    @property
    def summed_intensities(self) -> np.array:
        return sum(np.sum(ints) for ints in self.frame_ints)

    @property
    def num_peaks(self) -> np.array:
        return sum(len(ints) for ints in self.frame_ints)

    @staticmethod
    def generate_dummy(
        num_scans: int,
        offset: int = 0,
    ) -> tuple[FrameData, int]:
        frame_tofs = []
        frame_ints = []
        frame_counts = []
        count = 0
        for _scan in range(num_scans):
            count += 1
            frame_tofs.append(np.arange(1 + offset, 1 + offset + count))
            frame_ints.append(np.arange(1 + offset, 1 + offset + count) * 2)
            frame_counts.append(count)
            offset += count

        return (
            FrameData(
                frame_counts=np.array(frame_counts),
                frame_tofs=frame_tofs,
                frame_ints=frame_ints,
            ),
            offset,
        )

    @classmethod
    def from_array_lists(
        cls, tof_indices: list[np.array], int_arrays: list[np.array]
    ) -> FrameData:
        frame_counts = np.array([len(tof) for tof in tof_indices])
        for i, tof in enumerate(tof_indices):
            if not is_sorted_asc(tof):
                msg = f"Frame {i} TOF indices must be sorted in ascending order."
                raise ValueError(msg)

            if not len(tof) == len(int_arrays[i]):
                msg = f"Frame {i} TOF indices and intens. must have the same length."
                raise ValueError(msg)

        return cls(
            frame_counts=frame_counts, frame_tofs=tof_indices, frame_ints=int_arrays
        )

    @classmethod
    def from_value_arrays(
        cls,
        mzs: np.array,
        intensities: np.array,
        imss: np.array,
        ims_converter: SpecConverter,
        mz_converter: Tof2MzConverter,
        num_scans: int,
    ) -> FrameData:
        """Converts mz, intensity, and imss arrays to FrameData.

        Pseudo code:
        >   First convert imss to index.
        >   Then sort by ims.
        >       For every unique ims value:
        >           Find the indices that have that ims value.
        >           Sort those indices by mz.
        >           convert to tof index.
        >           Aggregate intensity on unique tof indices.
        >           Store in list.
        """
        if len(imss) == 0:
            return cls(
                frame_counts=np.array([0] * num_scans),
                frame_tofs=[],
                frame_ints=[],
            )

        im_indices = ims_converter.to_index(imss)
        order_im_indices = np.argsort(im_indices)
        sorted_im_indices = im_indices[order_im_indices]
        sorted_mzs = mzs[order_im_indices]
        sorted_intensities = intensities[order_im_indices]

        frame_counts = []
        frame_tofs = []
        frame_ints = []

        break_points = (np.where(np.diff(sorted_im_indices) != 0)[0] + 1).tolist()
        break_points = [0] + break_points + [len(sorted_im_indices)]

        last_scan_number = 0
        for i, bp in enumerate(zip(break_points[:-1], break_points[1:], strict=True)):
            start, end = bp
            curr_scan_number = sorted_im_indices[start]
            while (last_scan_number) < (curr_scan_number - 1):
                last_scan_number += 1
                frame_counts.append(0)
            tof_indices = mz_converter.to_index(sorted_mzs[start:end])
            unique_tofs = np.unique(tof_indices, return_inverse=True)
            uta = unique_tofs[0]
            if not is_sorted_asc(uta):
                msg = f"Frame {i} TOF indices must be sorted in ascending order."
                raise ValueError(msg)

            frame_tofs.append(uta)
            frame_ints.append(
                np.bincount(
                    unique_tofs[1], weights=sorted_intensities[start:end]
                ).astype("uint32")
            )
            curr_num_peaks = len(uta)
            frame_counts.append(curr_num_peaks)
            last_scan_number += 1

        overhang_scans = num_scans - last_scan_number
        frame_counts += [0] * overhang_scans
        out = cls(
            frame_counts=np.array(frame_counts),
            frame_tofs=frame_tofs,
            frame_ints=frame_ints,
        )

        return out

    def pack_data(self, frame_offset: int) -> bytes:
        local_frame_data = []

        # frame_offsets.append(frame_offset)
        scans = self.frame_counts
        scan_count = len(scans)
        ints = np.concatenate(self.frame_ints)
        # summed_intensities.append(np.sum(ints))
        # max_intensities.append(np.max(ints))
        # num_peaks.append(len(ints))
        buffer = np.zeros(scan_count + len(ints) * 2, dtype=np.uint32)
        buffer[0] = scan_count
        buffer[1:scan_count] = np.array(scans[:-1]) * 2
        buffer[scan_count + 1 :: 2] = ints
        offset = scan_count
        for tofs in self.frame_tofs:
            buffer[offset] = tofs[0]
            buffer[offset + 2 : offset + 2 * len(tofs) : 2] = np.diff(tofs)
            offset += 2 * len(tofs)
        buffer = np.frombuffer(buffer, dtype=np.uint8)
        buffer = buffer.reshape(-1, 4).T.flatten()
        decompressed_bytes = buffer
        compressed_data = pyzstd.compress(decompressed_bytes)
        compressed_data = np.frombuffer(compressed_data, dtype=np.uint8)
        frame_size = len(compressed_data) + 8
        local_frame_data.append(
            np.frombuffer(np.array([frame_size], dtype=np.uint32), dtype=np.uint8)
        )
        local_frame_data.append(
            np.frombuffer(np.array([scan_count], dtype=np.uint32), dtype=np.uint8)
        )
        local_frame_data.append(compressed_data)

        new_frame_offset = frame_offset + frame_size
        local_frame_data = np.concatenate(local_frame_data)
        out = local_frame_data.tobytes()

        return out, new_frame_offset


@dataclass
class FrameInfoBuilder:
    tdf_config: TDFConfig
    run_config: RunConfig

    def build_frames_df_template(self) -> pd.DataFrame:
        num_frames = self.run_config.num_cycles * self.run_config.frames_per_cycle
        frames_per_cycle = self.run_config.frames_per_cycle
        scanmode = self.tdf_config.SCAN_MODE
        num_scans = self.tdf_config.NUM_SCANS
        accum_time = self.tdf_config.ACCUMULATION_TIME
        ramp_time = self.tdf_config.RAMP_TIME

        frame_time = max(accum_time, ramp_time)
        frame_time_seconds = frame_time / 1000

        msms_type = [
            0 if i % frames_per_cycle == 0 else scanmode for i in range(num_frames)
        ]

        # This line is in the reference implementation but does nothing...
        # peaks = num_scans * (num_scans + 1) // 2

        frames = pd.DataFrame(
            {
                "Id": np.arange(1, num_frames + 1),
                "Time": np.arange(1, num_frames + 1, dtype=np.float64)
                * frame_time_seconds,
                "Polarity": ["+"] * num_frames,
                "ScanMode": [scanmode] * num_frames,
                "MsMsType": msms_type,
                "NumScans": [num_scans] * num_frames,
                "AccumulationTime": [accum_time] * num_frames,
                "RampTime": [ramp_time] * num_frames,
            }
        )
        return frames

    def build_frames_df(
        self,
        frame_offsets: np.array,
        max_intensities: np.array,
        summed_intensities: np.array,
        num_peaks: np.array,
    ) -> pd.DataFrame:
        msms_type = self.tdf_config.MSMS_TYPE

        # This line is in the reference implementation but does nothing...
        # peaks = num_scans * (num_scans + 1) // 2

        frames = self.build_frames_df_template()
        return self.complete_frames_df(
            frames,
            frame_offsets,
            max_intensities,
            summed_intensities,
            num_peaks,
        )

    def complete_frames_df(
        self,
        frames: pd.DataFrame,
        frame_offsets: np.array,
        max_intensities: np.array,
        summed_intensities: np.array,
        num_peaks: np.array,
    ) -> pd.DataFrame:
        frames["TimsId"] = frame_offsets
        frames["MaxIntensity"] = max_intensities
        frames["SummedIntensities"] = summed_intensities
        frames["NumPeaks"] = num_peaks

        return frames

    def build_dia_frame_msms_windows(self):
        num_dia_window_groups = self.run_config.num_dia_window_groups
        scan_groups_per_window_group = self.run_config.scan_groups_per_window_group

        dia_frame_msms_windows = pd.DataFrame(
            {
                "WindowGroup": [
                    1 + (x // scan_groups_per_window_group)
                    for x in range(num_dia_window_groups * scan_groups_per_window_group)
                ],
                "ScanNumBegin": [
                    30 + (200 * (x % scan_groups_per_window_group))
                    for x in range(num_dia_window_groups * scan_groups_per_window_group)
                ],
                "ScanNumEnd": 0,
                "IsolationMz": [
                    200 * (x + scan_groups_per_window_group)
                    for x in range(num_dia_window_groups * scan_groups_per_window_group)
                ],
                "IsolationWidth": 50,
                "CollisionEnergy": 42,
            }
        )
        dia_frame_msms_windows["ScanNumEnd"] = (
            150 + dia_frame_msms_windows["ScanNumBegin"]
        )
        return dia_frame_msms_windows

    def build_frame_msms_info(
        self, frames_id: pd.Series, frames_msms_type: pd.Series
    ) -> pd.DataFrame:
        num_dia_window_groups = self.run_config.num_dia_window_groups

        dia_frame_msms_info = pd.DataFrame(
            {
                "Frame": frames_id[frames_msms_type == 9],
                "WindowGroup": [
                    1 + (x // num_dia_window_groups)
                    for x in range((frames_msms_type == 9).sum())
                ],
            }
        )

        return dia_frame_msms_info

    def build_frame_msms_window_grops(self):
        num_dia_window_groups = self.run_config.num_dia_window_groups

        dia_frame_msms_window_groups = pd.DataFrame(
            {"id": [x + 1 for x in range(num_dia_window_groups)]}
        )
        return dia_frame_msms_window_groups
