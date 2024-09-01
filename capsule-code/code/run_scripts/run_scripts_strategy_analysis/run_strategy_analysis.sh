mkdir -p results/cnn_rgb/paper_plots/fig3/strategy_analysis/
mkdir -p results/cnn_rgb/paper_plots/fig4/strategy_analysis/
mkdir -p results/cnn_hs/paper_plots/fig3/strategy_analysis/
mkdir -p results/cnn_hs/paper_plots/fig4/strategy_analysis/

python code/Strategy_Analysis_Visualization/create_strategy_analysis_plots.py --fp-cams results/model_checkpoints/cnn_rgb/cv/default_cv2/gradcams/ --fp-testsplit code/Plant_Phenotyping/rgb_dataset_splits/test_2.txt --dtype rgb --config default
python code/Strategy_Analysis_Visualization/create_strategy_analysis_plots.py --fp-cams results/model_checkpoints/cnn_rgb/cv/rrr_cv2/gradcams/ --fp-testsplit code/Plant_Phenotyping/rgb_dataset_splits/test_2.txt --dtype rgb --config rrr
python code/Strategy_Analysis_Visualization/create_strategy_analysis_plots.py --fp-cams results/cnn_hs/gradcams_npy/default/ --fp-testsplit code/Plant_Phenotyping/hs_dataset_splits/cv1/val.txt --dtype hs --config default
python code/Strategy_Analysis_Visualization/create_strategy_analysis_plots.py --fp-cams results/cnn_hs/gradcams_npy/xil/ --fp-testsplit code/Plant_Phenotyping/hs_dataset_splits/cv1/val.txt --dtype hs --config rrr