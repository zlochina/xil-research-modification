#!/bin/bash

echo "running lagrange_constraint_optimisation.py..."

if [ -z "$1" ]; then
  # No argument passed
  poetry run python3 -m xil_research_modification.experiments.lagrange_constraint_optimisation \
    --config xil_research_modification/experiments/config_lagrange.yaml
else
  # Argument passed
  poetry run python3 -m xil_research_modification.experiments.lagrange_constraint_optimisation \
    --config xil_research_modification/experiments/config_lagrange.yaml --config_case "$1"
fi