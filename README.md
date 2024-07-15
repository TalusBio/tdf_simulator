
# Simulate TDF data.

```shell
TDF_SIMULATOR_DEBUG=1 LOGURU_LEVEL="TRACE" python  tdf_simulator/transition_simulator.py
# This will generate a dummy tdf with just increasing intensities
# Along the ion mobility axis
```

## Inspiration

The original implementation was adopted from the tests in timsrust,
which was written by Sander Willems
