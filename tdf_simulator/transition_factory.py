from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tdf_simulator.config import RunConfig, TDFConfig
from tdf_simulator.transition_simulator import (
    TransitionBundleSimulator,
    TransitionSimulator,
)


@dataclass
class TransitionSimulatorFactory:
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

    def build(self, *args, **kwargs) -> TransitionSimulator:
        kwargs = {**self.defaults, **kwargs}
        return TransitionSimulator(*args, **kwargs)

    def build_transition_bundle(
        self,
        kwargs_iter: Iterable[dict],
    ) -> TransitionBundleSimulator:
        lst = []
        for kwargs in kwargs_iter:
            lst.append(self.build(**kwargs))

        return TransitionBundleSimulator(lst, max_fwhms=self.rt_fwhm_edge)

    @classmethod
    def from_toml_config(cls, config: dict) -> TransitionSimulatorFactory:
        tdf_config = TDFConfig(**config["tdf_config"])
        run_config = RunConfig(**config["run_config"])
        defaults = config.get("defaults", None)
        rt_fwhm_edge = config.get("rt_fwhm_edge", 5.0)

        return cls(tdf_config, run_config, defaults, rt_fwhm_edge)
