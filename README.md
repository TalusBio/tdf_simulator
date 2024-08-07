
# Simulate TDF data.

```shell
TDF_SIMULATOR_DEBUG=1 LOGURU_LEVEL="TRACE" python  tdf_simulator/transition_simulator.py
# This will generate a dummy tdf with just increasing intensities
# Along the ion mobility axis
```

## Using a real file as a template

```shell
python -m tdf_simulator.config /path/to/my/dotd.d/analysis.tdf --toml settings.toml
```

## Inspiration

The original implementation was adopted from the tests in timsrust,
which was written by Sander Willems


### UNIMPLEMENTED FAKE ASPIRATIONAL API

It would be cool if we get to this point ... where
- a text file with peptides and intensities
- a tdf file that will be used to extract isolation window+tims parameters
- a config.toml that has parameters for the simulation (noise levels, FWHMs ...)

```
tdf_simulate --template_tdf ./mysupercoolrun.d/analysis.tdf --config config.toml --intensities peptides.txt
```
