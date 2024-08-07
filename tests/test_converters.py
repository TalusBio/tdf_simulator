import numpy as np

from tdf_simulator.converters import SpecConverter, Tof2MzConverter


def test_tof_convert_inplace() -> None:
    """Checks that in-place conversion works.

    (it actually modifies the input array)
    """
    conv = Tof2MzConverter.new(1, 100, 100)
    tof = np.array([1, 2, 3, 4, 5], dtype="float64")
    tof_copy = tof.copy()
    conv1 = conv.convert(tof)
    conv.convert_inplace(tof_copy)

    assert not np.allclose(tof_copy, tof)
    assert np.allclose(conv1, tof_copy)


def test_tof_back_and_forth() -> None:
    """Checks that the conversion back and forth is consistent."""
    conv = Tof2MzConverter.new(1, 100, 100)
    tof = np.array([1, 2, 3, 4, 5], dtype="float64")
    mz = conv.convert(tof)
    tof_back = conv.to_index(mz)

    assert np.allclose(tof, tof_back)


def test_spec_convert_inplace() -> None:
    """Checks that in-place conversion works.

    (it actually modifies the input array)
    """
    # slope: -0.0008737864077669902, intercept: 1.45
    conv = SpecConverter(intercept=1.45, slope=-0.0008737864077669902)
    spec = np.array([1, 2, 3, 4, 5], dtype="float64")
    spec_copy = spec.copy()
    conv1 = conv.convert(spec)
    conv.convert_inplace(spec_copy)

    assert not np.allclose(spec_copy, spec)
    assert np.allclose(conv1, spec_copy)


def test_spec_back_and_forth() -> None:
    """Checks that the conversion back and forth is consistent."""
    # slope: -0.0008737864077669902, intercept: 1.45
    conv = SpecConverter(intercept=1.45, slope=-0.0008737864077669902)
    scan = np.array([1, 2, 3, 4, 5], dtype="float64")
    ims = conv.convert(scan)
    scan_back = conv.to_index(ims)

    assert np.allclose(scan, scan_back)
