#!/bin/bash

set -e
set -o pipefail
set -o nounset

infile=$1
toml_file=tmp_coverage.toml
out_dotd=tmp_coverage.d

coverage run -m tdf_simulator.config $infile --toml $toml_file
coverage run --append -m tdf_simulator.main \
    --output_file $out_dotd \
    --config_file tests/data/settings.toml \
    --transitions_config_file  tests/data/transitions.json

rm -rf $out_dotd
coverage report --show-missing
