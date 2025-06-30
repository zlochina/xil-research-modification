#!/bin/bash

echo "running caipi_grid_search.py..."

if [ -z "$1" ]; then
  # No argument passed
  poetry run python3 -m xil_research_modification.experiments.caipi_grid_search \
    --config xil_research_modification/experiments/config.yaml
else
  # Argument passed
  poetry run python3 -m xil_research_modification.experiments.caipi_grid_search \
    --config xil_research_modification/experiments/config.yaml --config_case "$1"
fi