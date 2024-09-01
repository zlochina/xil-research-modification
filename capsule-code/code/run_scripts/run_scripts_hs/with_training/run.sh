#!/usr/bin/env bash

#rm -r results/model_checkpoints/cnn_hs
#rm -r results/cnn_hs

export PYTHONPATH=./code
export DATAPATH=data/plants_hs

bash code/run_scripts/run_scripts_hs/with_training/run_train_default_cv.sh
bash code/run_scripts/run_scripts_hs/with_training/run_train_xil_cv.sh

bash code/run_scripts/run_scripts_hs/with_training/run_evaluation_default.sh
bash code/run_scripts/run_scripts_hs/with_training/run_evaluation_xil.sh
bash code/run_scripts/run_scripts_hs/with_training/copy_evaluation.sh

bash code/run_scripts/run_scripts_hs/with_training/run_gradcam_default_cv.sh
bash code/run_scripts/run_scripts_hs/with_training/run_gradcam_xil_cv.sh
bash code/run_scripts/run_scripts_hs/with_training/copy_figure_explanations.sh

