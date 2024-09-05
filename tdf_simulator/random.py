import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import tomli

from tdf_simulator.config import WindowInfo
from tdf_simulator.simulator import TDFSimulator
from tdf_simulator.transition_factory import TransitionSimulatorFactory


@dataclass
class RandomTransitionBuilder:  # noqa: D101
    min_mz: float = 400
    max_mz: float = 1000
    min_time: float = 100
    max_time: float = 22 * 60
    min_charge = 2
    max_charge = 3
    envelope_intensities = [0.7, 0.3, 0.1]
    min_intensity: float = 1000
    max_intensity: float = 100_000
    min_apex_ims: float = 0.5
    max_apex_ims: float = 1.5
    ims_fwhm: float = 0.05
    rt_fwhm: float = 0.4
    # min_envelope_intensities: float = 0.1
    # For ms2 use only one envelope with intensity 1.0
    min_ms2_charge: int = 1
    max_ms2_charge: int = 3
    min_ms2_mz: float = 200
    max_ms2_mz: float = 1500
    num_transitions_min: int = 2
    num_transitions_max: int = 10

    def build_one(self) -> dict:  # noqa: D102
        ms2_mzs = [
            random.uniform(self.min_ms2_mz, self.max_ms2_mz)
            for _ in range(
                random.randint(self.num_transitions_min, self.num_transitions_max)
            )
        ]
        return {
            "ms1_mz": random.uniform(self.min_mz, self.max_mz),
            "ms1_charge": random.randint(self.min_charge, self.max_charge),
            "ms1_intensity": random.uniform(self.min_intensity, self.max_intensity),
            "apex_ims": random.uniform(self.min_apex_ims, self.max_apex_ims),
            "ims_fwhm": self.ims_fwhm,
            "apex_time": random.uniform(self.min_time, self.max_time),
            "time_fwhm": self.rt_fwhm,
            "ms2_mzs": ms2_mzs,
            "ms2_intensities": [
                random.uniform(self.min_intensity, self.max_intensity)
                for _ in range(len(ms2_mzs))
            ],
            "ms2_charges": [
                random.randint(self.min_ms2_charge, self.max_ms2_charge)
                for _ in range(len(ms2_mzs))
            ],
        }

    def build_transitions(self, num_transitions: int) -> list[dict]:  # noqa: D102
        return [self.build_one() for _ in range(num_transitions)]


def build_parser() -> argparse.ArgumentParser:  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Simulate a TDF files using random transitions."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        help="The path to the output file.",
        default="dia_test.d",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="The path to the config file. (toml)",
        default="config.toml",
    )
    parser.add_argument(
        "--num_transitions",
        type=int,
        help="The number of transitions to generate.",
        default=200,
    )
    return parser


def main(args: argparse.Namespace) -> None:  # noqa: D103
    with open(args.config_file, "rb") as f:
        config = tomli.load(f)

    transition_simulator_factory = TransitionSimulatorFactory.from_toml_config(config)
    tdf_config = transition_simulator_factory.tdf_config
    run_config = transition_simulator_factory.run_config
    windows = WindowInfo.from_toml_file(args.config_file)

    random_transition_builder = RandomTransitionBuilder()
    transitions_config = random_transition_builder.build_transitions(
        args.num_transitions
    )

    simulator_bundle = transition_simulator_factory.build_transition_bundle(
        transitions_config
    )
    simulator = TDFSimulator(
        folder_name=args.output_file,
        tdf_config=tdf_config,
        run_config=run_config,
        windows=windows,
    )
    simulator.peak_simulator = simulator_bundle
    if Path(args.output_file).exists():
        raise FileExistsError(f"Output file {args.output_file} already exists.")

    simulator.generate_data()


if __name__ == "__main__":
    parser = build_parser()
    args, unkargs = parser.parse_known_args()
    if unkargs:
        parser.error(f"Unknown arguments: {unkargs}")
    main(args)
