import os
from dataclasses import dataclass
from functools import cached_property
from typing import Union

import numpy as np
from loguru import logger
from uniplot import plot_to_string

from tdf_simulator.config import RunConfig, TDFConfig
from tdf_simulator.converters import SpecConverter, Tof2MzConverter
from tdf_simulator.frames import FrameData
from tdf_simulator.simulator import PeakSimulator, WindowInfo

# I know ... development ... ill look it up later
NEUTRON_MASS = 1.0


class GaussianSimulator:
    def __init__(
        self,
        mz,
        intensity,
        apex_time,
        time_fwhm,
        apex_ims,
        ims_fwhm,
    ):
        self.mz = mz
        self.intensity = intensity
        self.apex_time = apex_time
        self.time_fwhm = time_fwhm
        self.apex_ims = apex_ims
        self.ims_fwhm = ims_fwhm

    def sample_ratio_at_time(self, time: float, imms_vals: np.array) -> np.array:
        time_diff = time - self.apex_time
        time_factor = np.exp(-4 * np.log(2) * time_diff**2 / self.time_fwhm**2)
        imms_diff = imms_vals - self.apex_ims
        imms_factor = np.exp(-4 * np.log(2) * imms_diff**2 / self.ims_fwhm**2)
        return time_factor * imms_factor

    def sample_intensity_at_time(self, time: float, imms_vals: np.array) -> np.array:
        return self.intensity * self.sample_ratio_at_time(time, imms_vals)


class TransitionSimulator(PeakSimulator):
    THRESHOLD_INTENSITY = 10

    def __init__(
        self,
        ms1_mz: float,
        ms1_intensity: float,
        ms2_mzs: list[float],
        ms2_intensities: list[float],
        apex_time: float,
        time_fwhm: float,
        apex_ims: float,
        ims_fwhm: float,
        tdf_config: TDFConfig,
        run_config: RunConfig,
        envelope_intensities: list[float] | None = None,
        ms2_envelope_intensities: list[float] | None = None,
        ms2_charges: list[int] | None = None,
        ms1_charge: int = 1,
    ) -> None:
        self.ms1_gauss = GaussianSimulator(
            mz=ms1_mz,
            intensity=ms1_intensity,
            apex_time=apex_time,
            time_fwhm=time_fwhm,
            apex_ims=apex_ims,
            ims_fwhm=ims_fwhm,
        )

        if len(ms2_mzs) != len(ms2_intensities):
            raise ValueError("ms2_mzs and ms2_intensities must have the same length")

        self.ms2_intensities = np.array(ms2_intensities)
        self.ms2_mzs = np.array(ms2_mzs)

        self.tdf_config = tdf_config
        self.run_config = run_config
        self.ims_converter = SpecConverter.from_config(tdf_config)
        self.mz_converter = Tof2MzConverter.from_config(tdf_config)
        if envelope_intensities is None:
            envelope_intensities = [1]

        if ms2_envelope_intensities is None:
            ms2_envelope_intensities = [1]

        if ms2_charges is None:
            ms2_charges = [1] * len(ms2_mzs)

        self.ms2_charges = np.array(ms2_charges)
        self.ms1_charge = ms1_charge
        self.envelope_intensities = np.array(envelope_intensities)
        self.ms2_envelope_intensities = np.array(ms2_envelope_intensities)

    @property
    def apex_time(self) -> float:
        return self.ms1_gauss.apex_time

    @property
    def time_fwhm(self) -> float:
        return self.ms1_gauss.time_fwhm

    @cached_property
    def ims_values(self) -> np.array:
        return self.ims_converter.convert(
            np.arange(self.tdf_config.NUM_SCANS, dtype=np.float64)
        )

    def generate_frame_data(
        self,
        time: float,
        window_info: list[WindowInfo] | None,
    ) -> tuple[FrameData, int]:
        out = FrameData.from_value_arrays(
            **self._generate_frame_data(time, window_info),
            num_scans=self.tdf_config.NUM_SCANS,
        )
        npeaks = len(out.frame_ints)

        return out, npeaks

    def _generate_frame_data(
        self,
        time: float,
        window_info: list[WindowInfo] | None,
    ) -> dict[str, Union[np.array, TDFConfig, RunConfig]]:
        tmp = []
        if window_info is None:
            # MS1 frame
            tmp.append(self._generate_ms1_frame_data(time))
        else:
            for window in window_info:
                iso_low = window.isolation_mz - (window.isolation_width / 2)
                iso_high = window.isolation_mz + (window.isolation_width / 2)
                if self.ms1_gauss.mz < iso_low or self.ms1_gauss.mz > iso_high:
                    continue

                tmp.append(
                    self._generate_frame_data_ms2_window(
                        time, window.scan_num_begin, window.scan_num_end
                    )
                )

        if len(tmp) == 0:
            return {
                "mzs": np.array([]),
                "intensities": np.array([]),
                "imss": np.array([]),
                "ims_converter": self.ims_converter,
                "mz_converter": self.mz_converter,
            }

        out = self._remove_out_of_mz(
            mzs=np.concatenate([d["mzs"] for d in tmp]),
            intens=np.concatenate([d["intensities"] for d in tmp]),
            imss=np.concatenate([d["imss"] for d in tmp]),
        )

        out.update(
            {
                "ims_converter": self.ims_converter,
                "mz_converter": self.mz_converter,
            }
        )
        return out

    def _remove_out_of_mz(
        self, mzs: np.array, intens: np.array, imss: np.array
    ) -> dict[str, np.array]:
        keep = (mzs > self.tdf_config.MZ_MIN) & (mzs < self.tdf_config.MZ_MAX)
        return {
            "mzs": mzs[keep],
            "intensities": intens[keep],
            "imss": imss[keep],
        }

    def _generate_ms1_frame_data(self, time: float) -> tuple[FrameData, int]:
        intens = self.ms1_gauss.sample_intensity_at_time(
            time, np.expand_dims(self.ims_values, axis=0)
        )
        mzs = np.array([self.ms1_gauss.mz])

        intens_ems, imss_ems, mzs_ems = self._expand_isotopes(
            intens=intens,
            imss=self.ims_values,
            isotope_ratios=self.envelope_intensities,
            mzs=mzs,
            charges=np.array([self.ms1_charge]),
        )

        self.__plot(intens_ems, imss_ems, time)
        keep = intens_ems > self.THRESHOLD_INTENSITY
        return {
            "mzs": mzs_ems[keep],
            "intensities": intens_ems[keep],
            "imss": imss_ems[keep],
            "ims_converter": self.ims_converter,
            "mz_converter": self.mz_converter,
        }

    def _generate_frame_data_ms2_window(
        self, time: float, scan_start: int, scan_end: int
    ) -> tuple[FrameData, int]:
        window_imss = self.ims_values[scan_start:scan_end]
        # Note ... the compute happens before expanding the dimensions
        ratios = self.ms1_gauss.sample_ratio_at_time(time, window_imss)
        intens = np.einsum("r, i -> ir", ratios, self.ms2_intensities)
        # Done with intensive compute

        intens_ems, imss_ems, mzs_ems = self._expand_isotopes(
            intens=intens,
            imss=window_imss,
            isotope_ratios=self.ms2_envelope_intensities,
            mzs=self.ms2_mzs,
            charges=self.ms2_charges,
        )

        self.__plot(intens_ems, imss_ems, time)
        keep = intens_ems > self.THRESHOLD_INTENSITY
        return {
            "mzs": mzs_ems[keep],
            "intensities": intens_ems[keep],
            "imss": imss_ems[keep],
            "ims_converter": self.ims_converter,
            "mz_converter": self.mz_converter,
        }

    @staticmethod
    def __plot(intens: np.array, imss: np.array, time: float) -> None:
        if os.environ.get("TDF_SIMULATOR_DEBUG"):
            plot = plot_to_string(
                intens,
                imss,
                title=f"Transition frame at time {time}",
            )
            logger.debug("\n" + ("\n".join(plot)))

    def _expand_isotopes(
        self,
        intens: np.array,
        imss: np.array,
        isotope_ratios: np.array,
        mzs: np.array,
        charges: np.array,
    ) -> np.array:
        logger.trace("Expanding isotopes")
        logger.trace(f"intens: {intens.shape}")
        logger.trace(f"imss: {imss.shape}")
        logger.trace(f"isotope_ratios: {isotope_ratios.shape}")
        logger.trace(f"mzs: {mzs.shape}")
        logger.trace(f"charges: {charges.shape}")

        # If isotopes ratio has shape (e,) for the number of isotopes.
        # And intens has shape (m, s), for the number of mzs and scans.
        # And imss has shape (s,) for the number of scans.
        # And charges and mzs have shape (m,) for the number of mzs...

        # This should return:
        # - intensities with shape (m * e, s)
        # - imss with shape (m * e, s)
        # - mzs with shape (m * e, s)

        # check shapes to make sure I didnt break up anything upstream...
        if not charges.shape == mzs.shape:
            raise ValueError(
                "All arrays must have the same length"
                f"Got {charges.shape} for charge and {mzs.shape} for mzs"
            )

        if not len(isotope_ratios.shape) == 1:
            raise ValueError(
                f"Isotope ratios must have shape (e,), got shape {isotope_ratios.shape}"
            )

        if not len(intens.shape) == 2:
            raise ValueError(
                "Intens must have shape (m, s), m for mzs and s for scans, "
                f"Got shape {intens.shape}"
            )

        if not len(imss.shape) == 1:
            raise ValueError(f"Imss must have shape (s,), Got shape {imss.shape}")

        if not len(mzs.shape) == 1:
            raise ValueError(f"Mzs must have shape (m,), got shape {mzs.shape}")

        if not len(charges.shape) == 1:
            raise ValueError(f"Charges must have shape (m,), got shape {charges.shape}")

        # Now I can start the expansion
        # I will first using matrix multiplication to expand the intensities
        # with the isotope ratios

        intens_ems = np.einsum("e, ms -> ems", isotope_ratios, intens)

        # I will generate an offset vector for the mzs
        mz_offsets_e = np.arange(len(isotope_ratios), dtype=np.float64)
        mz_offsets_em = np.einsum("e, m -> em", mz_offsets_e, NEUTRON_MASS / charges)

        # I will expand the mzs
        mzs_em = (
            np.expand_dims(mzs, axis=0).repeat(len(isotope_ratios), axis=0)
            + mz_offsets_em
        )

        mz_ems = np.expand_dims(
            mzs_em,
            axis=2,
        ).repeat(len(imss), axis=2)

        # I will expand the imss
        imss_ems = np.expand_dims(
            np.expand_dims(imss, axis=0).repeat(len(mzs), axis=0), axis=0
        ).repeat(len(isotope_ratios), axis=0)

        # validate shapes
        if not intens_ems.shape == mz_ems.shape == imss_ems.shape:
            msg = "Shapes do not match"
            msg += f"intens_ems: {intens_ems.shape},"
            msg += f" mz_ems: {mz_ems.shape},"
            msg += f" imss_ems: {imss_ems.shape}"
            raise ValueError(msg)

        # When returning I will flatten the e-m dimensions.
        out = [arr.reshape(-1, arr.shape[-1]) for arr in (intens_ems, imss_ems, mz_ems)]
        return out


@dataclass
class TransitionBundleSimulator:
    elems: list[TransitionSimulator]
    max_fwhms: float
    _left_edge: np.array = None
    _right_edge: np.array = None
    _rollmax_right_edge: np.array = None

    def __post_init__(self):
        # Sort by apex - (max_fwhms * fwhm)
        self.elems.sort(key=lambda x: x.apex_time - (self.max_fwhms * x.time_fwhm))
        self._left_edge = np.array(
            [x.apex_time - (self.max_fwhms * x.time_fwhm) for x in self.elems]
        )
        # Note this is not necessarily sorted
        self._right_edge = np.array(
            [x.apex_time + (self.max_fwhms * x.time_fwhm) for x in self.elems]
        )
        # This one will be! Since its the rolling max
        self._rollmax_right_edge = np.maximum.accumulate(self._right_edge)

    def generate_frame_data(self, time: float, window_info: list[WindowInfo] | None):
        array_dicts = self._generate_frame_data(time, window_info)
        out = FrameData.from_value_arrays(
            **array_dicts,
            num_scans=self.elems[0].tdf_config.NUM_SCANS,
        )
        npeaks = len(out.frame_ints)
        return out, npeaks

    def _generate_frame_data(self, time: float, window_info: list[WindowInfo] | None):
        right_end = np.searchsorted(self._left_edge, time, "right")
        left_start = np.searchsorted(self._right_edge, time, "left")

        tmp = []
        for elem in self.elems[left_start:right_end]:
            tmp.append(elem._generate_frame_data(time, window_info))

        if len(tmp) == 0:
            return {
                "mzs": np.array([]),
                "intensities": np.array([]),
                "imss": np.array([]),
                "ims_converter": self.elems[0].ims_converter,
                "mz_converter": self.elems[0].mz_converter,
            }

        out = {
            "mzs": np.concatenate([d["mzs"] for d in tmp]),
            "intensities": np.concatenate([d["intensities"] for d in tmp]),
            "imss": np.concatenate([d["imss"] for d in tmp]),
        }

        out.update(
            {
                "ims_converter": self.elems[0].ims_converter,
                "mz_converter": self.elems[0].mz_converter,
            }
        )

        return out
