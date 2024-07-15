import numpy as np

from tdf_simulator.config import RunConfig, TDFConfig
from tdf_simulator.transition_simulator import (
    GaussianSimulator,
    TransitionSimulator,
    WindowInfo,
)


def test_gaussian_simulator() -> None:
    """Tests the Gaussian simulator."""
    gauss = GaussianSimulator(
        mz=100,
        intensity=1000,
        apex_time=60,
        time_fwhm=1,
        apex_ims=0,
        ims_fwhm=1,
    )
    time = 0
    imms_vals = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    intens = gauss.sample_intensity_at_time(time, imms_vals)
    assert np.allclose(intens, [0, 0, 0, 0, 0])

    time = 60
    intens = gauss.sample_intensity_at_time(time, imms_vals)
    assert np.allclose(intens, [1000, 62.50, 0.01, 0, 0], atol=0.01, rtol=0.5)

    time = 61
    intens = gauss.sample_intensity_at_time(time, imms_vals)
    assert np.allclose(intens, [62.5, 3.906, 0, 0, 0], atol=0.01, rtol=0.5)

    time = 59
    intens = gauss.sample_intensity_at_time(time, imms_vals)
    assert np.allclose(intens, [62.5, 3.906, 0, 0, 0], atol=0.01, rtol=0.5)


def test_transition_simulator() -> None:
    """Tests the Transition simulator."""
    tdf_config = TDFConfig(
        MZ_MIN=100,
        MZ_MAX=1000,
    )
    run_config = RunConfig(
        num_cycles=2,
        frames_per_cycle=3,
        num_dia_window_groups=2,
        scan_groups_per_window_group=2,
    )

    ms1_mz = 100
    ms1_intensity = 1000
    ms2_mzs = [100.5, 200, 300, 998.5]
    ms2_intensities = [1000, 2000, 3000, 4000]
    apex_time = 60
    time_fwhm = 1
    apex_ims = 1.1
    ims_fwhm = 0.1

    ts = TransitionSimulator(
        ms1_mz=ms1_mz,
        ms1_intensity=ms1_intensity,
        ms2_mzs=ms2_mzs,
        ms2_intensities=ms2_intensities,
        apex_time=apex_time,
        time_fwhm=time_fwhm,
        apex_ims=apex_ims,
        ims_fwhm=ims_fwhm,
        tdf_config=tdf_config,
        run_config=run_config,
    )

    wi = WindowInfo(
        window_group=1,
        scan_num_begin=300,
        scan_num_end=400,
        isolation_mz=100,
        isolation_width=50,
        collision_energy=42,
    )

    frame_data = ts._generate_frame_data(60, [wi])
    unique_out_mzs = np.unique(frame_data["mzs"], return_inverse=True)
    assert np.allclose(unique_out_mzs[0], [100.5, 200, 300, 998.5])
    for i, m in enumerate(unique_out_mzs[0]):
        # This will catch if accidentally I am multiplying intensities to numbers higher
        # that the set intensity.
        assert (
            np.max(frame_data["intensities"][frame_data["mzs"] == m])
            < ms2_intensities[i]
        )

    ## Test window out of ms1 mz
    wi_out = WindowInfo(
        window_group=1,
        scan_num_begin=300,
        scan_num_end=400,
        isolation_mz=20_000,
        isolation_width=50,
        collision_energy=42,
    )
    frame_data1 = ts.generate_frame_data(60, [wi])
    frame_data2 = ts.generate_frame_data(60, [wi_out])

    assert len(frame_data2[0].frame_ints) == 0
    assert len(frame_data1[0].frame_ints) > 0
    assert len(frame_data1[0].frame_ints) == frame_data1[1]

    # frame_data_ms1 = ts.generate_frame_data(60, None)


def test_isotope_transition_sim():
    tdf_config = TDFConfig(
        MZ_MIN=100,
        MZ_MAX=1000,
    )
    run_config = RunConfig(
        num_cycles=2,
        frames_per_cycle=3,
        num_dia_window_groups=2,
        scan_groups_per_window_group=2,
    )

    ms1_mz = 110
    ms1_intensity = 1000
    ms2_mzs = [100, 200, 300, 998.45]
    ms2_intensities = [1000, 2000, 3000, 4000]
    apex_time = 60
    time_fwhm = 1
    apex_ims = 1.1
    ims_fwhm = 0.1

    ts = TransitionSimulator(
        ms1_mz=ms1_mz,
        ms1_intensity=ms1_intensity,
        ms2_mzs=ms2_mzs,
        ms2_intensities=ms2_intensities,
        apex_time=apex_time,
        time_fwhm=time_fwhm,
        apex_ims=apex_ims,
        ims_fwhm=ims_fwhm,
        tdf_config=tdf_config,
        run_config=run_config,
        envelope_intensities=[0.1, 0.2, 0.3],
        ms2_charges=[1, 2, 3, 2],
        ms1_charge=1,
    )

    wi = WindowInfo(
        window_group=1,
        scan_num_begin=300,
        scan_num_end=400,
        isolation_mz=100,
        isolation_width=50,
        collision_energy=42,
    )
    wi_out = WindowInfo(
        window_group=1,
        scan_num_begin=300,
        scan_num_end=400,
        isolation_mz=20_000,
        isolation_width=50,
        collision_energy=42,
    )
    frame_data1 = ts.generate_frame_data(60, [wi])
    assert len(frame_data1[0].frame_ints) > 0
    num_unique_mzs = len(np.unique(np.concatenate(frame_data1[0].frame_tofs)))
    assert (
        num_unique_mzs > 9
    ), f"Expected 12 (>9 is ok ..) unique mzs, got {num_unique_mzs}"

    frame_data2 = ts.generate_frame_data(60, [wi_out])
    assert frame_data2[1] == 0

    frame_data_ms1 = ts.generate_frame_data(60, None)
    num_unique_mzs = len(np.unique(np.concatenate(frame_data_ms1[0].frame_tofs)))
    assert num_unique_mzs == 3, f"Expected 3 unique mzs, got {num_unique_mzs}"


if __name__ == "__main__":
    test_gaussian_simulator()
    test_transition_simulator()
    test_isotope_transition_sim()
    print("All tests passed!")
