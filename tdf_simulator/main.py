from tdf_simulator.config import RunConfig, TDFConfig, WindowInfo
from tdf_simulator.simulator import TDFSimulator
from tdf_simulator.transition_factory import TransitionSimulatorFactory
import argparse
from pathlib import Path
import tomli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate a TDF files.")

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
        "--transitions_config_file",
        type=str,
        help="The path to the config file. (json)",
        default="transitions.json",
    )
    parser.add_argument(
        "--write_config_template",
        type=str,
        help=(
            "Write the config template to the output file. "
            "If passed it will only generage "
            "the template and exit."
        ),
    )
    return parser


def main(args: argparse.Namespace) -> None:
    if args.write_config_template:
        if Path(args.output_file).exists():
            raise FileExistsError(f"Output file {args.output_file} already exists.")
        if not args.write_config_template.endswith(".toml"):
            msg = "The write_config_template must end with .toml, "
            msg += f" got {args.write_config_template}"
            raise ValueError(msg)
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        tdf_config = TDFConfig()
        run_config = RunConfig(
            num_cycles=2,
            frames_per_cycle=3,
            num_dia_window_groups=2,
            scan_groups_per_window_group=2,
        )

    with open(args.config_file, "rb") as f:
        config = tomli.load(f)

    transition_simulator_factory = TransitionSimulatorFactory.from_toml_config(config)
    tdf_config = transition_simulator_factory.tdf_config
    run_config = transition_simulator_factory.run_config
    windows = WindowInfo.from_toml_file(args.config_file)

    simulator_bundle = transition_simulator_factory.build_transition_bundle_from_json(
        args.transitions_config_file
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
