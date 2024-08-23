from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass
from sqlite3 import Connection

import numpy as np
from loguru import logger

from tdf_simulator.config import TDFConfig


@dataclass
class SpecConverter:
    """Converts scan indices to 1/k0 values and vice versa."""

    intercept: float
    slope: float

    def convert(self, num: np.typing.ArrayLike) -> np.array:
        """Converts a numpy array of scan indices to 1/k0 values."""
        return (num * self.slope) + self.intercept

    def convert_inplace(self, num: np.typing.ArrayLike) -> None:
        """Converts a numpy array of scan indices to 1/k0 values in place.

        Note: This assumes the array is already a float array.
        There is no type checking for performance reasons. So be careful.
        """
        num *= self.slope
        num += self.intercept

    def to_index(self, num: np.typing.ArrayLike, *, safely: bool = True) -> np.array:
        """Converts a numpy array of 1/k0 values to scan indices."""
        out = ((num - self.intercept) / self.slope).astype(np.int64)

        if safely and np.any(out < 0):
            raise ValueError("Some values are below 0. This is not allowed.")
        return out

    @classmethod
    @lru_cache(maxsize=32)
    def from_params(
        cls,
        im_min: float,
        im_max: float,
        scan_max_index: int,
    ) -> SpecConverter:
        """Creates a SpecConverter from the given parameters.

        These parameters can be read from the .tdf file and internally come
        from the TDFConfig object.
        """
        scan_intercept = im_max
        scan_slope = (im_min - scan_intercept) / scan_max_index

        logger.info(
            f"Starting SpecConverter, with slope: {scan_slope}"
            f", intercept: {scan_intercept}",
        )

        return cls(slope=scan_slope, intercept=scan_intercept)

    @classmethod
    def from_config(cls, config: TDFConfig) -> SpecConverter:
        """Creates a SpecConverter from a TDFConfig object."""
        return cls.from_params(
            im_min=config.IM_MIN,
            im_max=config.IM_MAX,
            scan_max_index=config.NUM_SCANS,
        )

    @classmethod
    def from_connection(cls, conn: Connection) -> SpecConverter:
        """Creates a SpecConverter from a SQLite connection to a tdf file."""
        dat = conn.execute("SELECT Key, Value FROM GlobalMetadata;")
        # OneOverK0AcqRangeLower
        # OneOverK0AcqRangeUpper
        dat = {v[0]: v[1] for v in dat}

        im_min = float(dat["OneOverK0AcqRangeLower"])
        im_max = float(dat["OneOverK0AcqRangeUpper"])

        # This line feels flaky ... I am not sure if this piece of data
        # is stored elsewhere in the database.
        scan_max_index = max(x[0] for x in conn.execute("SELECT NumScans FROM Frames;"))
        return cls.from_params(im_min, im_max, scan_max_index)


@dataclass
class Tof2MzConverter:
    """Converts TOF indices to m/z values and vice versa."""

    tof_intercept: float
    tof_slope: float

    @classmethod
    def new(cls, mz_min: float, mz_max: float, tof_max_index: int) -> Tof2MzConverter:
        """Creates a new Tof2MzConverter object from the given parameters."""
        tof_intercept = math.sqrt(mz_min)
        tof_slope = (math.sqrt(mz_max) - tof_intercept) / tof_max_index

        logger.info(
            f"Starting Tof2MzConverter, with slope: {tof_slope}"
            f", intercept: {tof_intercept}"
        )
        return Tof2MzConverter(tof_intercept, tof_slope)

    def convert(self, tof_index_f64: np.array) -> np.array:
        """Converts a numpy array of TOF indices to m/z values."""
        return np.square(self.tof_intercept + self.tof_slope * tof_index_f64)

    def convert_inplace(self, tof_index_f64: np.array) -> None:
        """Converts a numpy array of TOF indices to m/z values in place.

        Note: This assumes the array is already a float array.
        There is no type checking for performance reasons. So be careful.
        """
        tof_index_f64 *= self.tof_slope
        tof_index_f64 += self.tof_intercept
        np.square(tof_index_f64, out=tof_index_f64)

    def to_index(self, mz_f64: np.array, *, safely: bool = True) -> np.array:
        """Converts a numpy array of m/z values to TOF indices."""
        out = ((np.sqrt(mz_f64) - self.tof_intercept) / self.tof_slope).astype(np.int64)
        if safely and np.any(out < 0):
            raise ValueError("Some values are below 0. This is not allowed.")

        return out

    @classmethod
    @lru_cache(maxsize=32)
    def from_config(cls, config: TDFConfig) -> Tof2MzConverter:
        """Creates a Tof2MzConverter from a TDFConfig object."""
        return cls.new(
            mz_min=config.MZ_MIN,
            mz_max=config.MZ_MAX,
            tof_max_index=config.NUM_TOF_BINS,
        )

    @staticmethod
    def from_connection(conn: Connection) -> Tof2MzConverter:
        """Creates a Tof2MzConverter from a SQLite connection to a tdf file."""
        cursor = conn.cursor()

        # Read TOF Max Index
        cursor.execute(
            "SELECT Value FROM GlobalMetadata WHERE Key = 'DigitizerNumSamples'"
        )
        tof_max_index_string = cursor.fetchone()[0]
        tof_max_index = int(tof_max_index_string)

        # Read MZ Max Value
        cursor.execute("SELECT Value FROM GlobalMetadata WHERE Key = 'MzAcqRangeUpper'")
        mz_max_value_string = cursor.fetchone()[0]
        mz_max_value = float(mz_max_value_string)

        # Read MZ Min Value
        cursor.execute("SELECT Value FROM GlobalMetadata WHERE Key = 'MzAcqRangeLower'")
        mz_min_value_string = cursor.fetchone()[0]
        mz_min_value = float(mz_min_value_string)

        logger.info(
            f"Starting Tof2MzConverter, with tof max ind: {tof_max_index}"
            f", max val: {mz_max_value}"
            f", mz min: {mz_min_value}",
        )

        return Tof2MzConverter.new(mz_min_value, mz_max_value, tof_max_index)
