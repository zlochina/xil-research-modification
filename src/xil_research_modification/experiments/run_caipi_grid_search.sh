poetry run python3 caipi_grid_search.py --threshold 0.95 \
 --output_filename caipi_experiment/caipi_grid_search_1run.csv \
 --current_path . --optimizer sgd --train_from_ground_zero \
 --evaluate_every_nth_epoch 10 --train_dataset_size 200