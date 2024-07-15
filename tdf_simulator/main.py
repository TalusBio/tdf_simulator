from tdf_simulator.config import RunConfig, TDFConfig
from tdf_simulator.simulator import TDFSimulator


def main():
    tdf_config = TDFConfig()
    run_config = RunConfig(
        num_cycles=2,
        frames_per_cycle=3,
        num_dia_window_groups=2,
        scan_groups_per_window_group=2,
    )
    simulator = TDFSimulator(
        folder_name="dia_test.d",
        tdf_config=tdf_config,
        run_config=run_config,
    )
    simulator.generate_data()


if __name__ == "__main__":
    main()
