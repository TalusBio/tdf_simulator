from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

from tdf_simulator.config import RunConfig, TDFConfig
from tdf_simulator.transition_simulator import (
    TransitionBundleSimulator,
    TransitionSimulator,
)


@dataclass
class TransitionSimulatorFactory:
    """Factory for creating TransitionSimulator objects."""

    tdf_config: TDFConfig
    run_config: RunConfig
    defaults: dict = None
    rt_fwhm_edge: float = 5.0

    def __post_init__(self):
        if self.defaults is None:
            self.defaults = {
                "tdf_config": self.tdf_config,
                "run_config": self.run_config,
            }

        else:
            self.defaults = {
                **self.defaults,
                "tdf_config": self.tdf_config,
                "run_config": self.run_config,
            }

    def build(self, *args, **kwargs) -> TransitionSimulator:  # noqa: ANN002,ANN003
        """Build a TransitionSimulator object.

        Args:
            *args: Arguments to pass to the TransitionSimulator constructor.
            **kwargs:
                Keyword arguments to pass to the TransitionSimulator constructor.
                More accurately, these are the arguments that will be passed to the
                TransitionSimulator constructor, but with the defaults from this
                factory applied.

        Returns:
            TransitionSimulator: A TransitionSimulator object.

        See Also:
            TransitionSimulator
        """
        kwargs = {**self.defaults, **kwargs}
        return TransitionSimulator(*args, **kwargs)

    def build_transition_bundle(
        self,
        kwargs_iter: Iterable[dict],
    ) -> TransitionBundleSimulator:
        """Build a TransitionBundleSimulator object.

        Args:
            kwargs_iter: An iterable of dictionaries. Each dictionary contains the
                keyword arguments to pass to the TransitionSimulator constructor.
                Check the TransitionSimulator constructor for the expected keyword
                arguments.

        Returns:
            TransitionBundleSimulator: A TransitionBundleSimulator object.

        """
        lst = []
        for kwargs in kwargs_iter:
            lst.append(self.build(**kwargs))

        return TransitionBundleSimulator(lst, max_fwhms=self.rt_fwhm_edge)

    def build_transition_bundle_from_json(
        self, json_path: str
    ) -> TransitionBundleSimulator:
        """Build a TransitionBundleSimulator object from a JSON file.

        Args:
            json_path (str): The path to the JSON file.

        Returns:
            TransitionBundleSimulator: A TransitionBundleSimulator object.
        """
        with open(json_path) as f:
            config = json.load(f)

        if "transitions" in config:
            config = config["transitions"]

        return self.build_transition_bundle(config)

    @classmethod
    def from_toml_config(cls, config: dict) -> TransitionSimulatorFactory:
        """Create a TransitionSimulatorFactory object from a TOML configuration.

        Args:
            config: A dictionary containing the TOML configuration.

        Details:
            The configuration must contain the following keys:
                - "tdf_config": A dictionary containing the TDFConfig configuration.
                - "run_config": A dictionary containing the RunConfig configuration.
            The configuration may also contain the following optional keys:
                - "defaults": A dictionary containing default values to pass to the
                    TransitionSimulator constructor.
                - "rt_fwhm_edge": A float representing the maximum FWHM for retention
                    time edges. (will not simulate intensities beyond this range for
                    each transition)

        Returns:
            TransitionSimulatorFactory: A TransitionSimulatorFactory object.

        """
        tdf_config = TDFConfig(**config["tdf_config"])
        run_config = RunConfig(**config["run_config"])
        defaults = config.get("defaults", None)
        # max_rt = run_config.num_cycles * tdf_config.bottleneck_time_ms / 1000
        # Max width a peak can have ... ish ...
        rt_fwhm_edge = config.get("rt_fwhm_edge", 5.0)

        return cls(tdf_config, run_config, defaults, rt_fwhm_edge)
