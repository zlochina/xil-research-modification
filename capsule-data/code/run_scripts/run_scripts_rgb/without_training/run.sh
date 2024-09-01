#!/bin/bash

export PYTHONPATH=./code
export DATAPATH=data/plant_rgb

# Evaluate default, generate gradcams
bash code/run_scripts/run_scripts_rgb/without_training/run_test_rgb_default.sh
bash code/run_scripts/run_scripts_rgb/without_training/run_gen_cams_rgb_default.sh

# Evaluate rrr, generate gradcams
bash code/run_scripts/run_scripts_rgb/without_training/run_test_rgb_rrr.sh
bash code/run_scripts/run_scripts_rgb/without_training/run_gen_cams_rgb_rrr.sh

# store examples of figure in separate folder
bash code/run_scripts/run_scripts_rgb/without_training/copy_figure_examples_explanations.sh