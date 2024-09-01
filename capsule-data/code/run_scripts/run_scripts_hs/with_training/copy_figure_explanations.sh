mkdir -p results/cnn_hs/paper_plots/fig1/xil/
mkdir -p results/cnn_hs/paper_plots/fig1/default/
mkdir -p results/cnn_hs/paper_plots/figures_wavelength_explanations/xil/
mkdir -p results/cnn_hs/paper_plots/figures_wavelength_explanations/default/
mkdir -p results/cnn_hs/paper_plots/fig3/
mkdir -p results/cnn_hs/paper_plots/fig4/
mkdir -p results/cnn_hs/gradcams_npy/xil/
mkdir -p results/cnn_hs/gradcams_npy/default/

cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai12_id2,Z18,4,1,1_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig1/xil/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai12_id2,Z17,1,0,0_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig1/xil/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai12_id2,Z16,2,1,1_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig1/xil/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai12_id2,Z15,2,1,2_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig1/xil/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai2_id2,Z8,4,0,0_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig1/xil/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai2_id2,Z8,4,1,2_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig1/xil/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai0_id2,Z1,3,1,1_class0_gt0_pred0_mean.jpg results/cnn_hs/paper_plots/fig1/xil/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai0_id2,Z2,1,0,2_class0_gt0_pred0_mean.jpg results/cnn_hs/paper_plots/fig1/xil/

cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai12_id2,Z18,4,1,1_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig1/default/
cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai12_id2,Z17,1,0,0_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig1/default/
cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai12_id2,Z16,2,1,1_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig1/default/
cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai12_id2,Z15,2,1,2_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig1/default/
cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai2_id2,Z8,4,0,0_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig1/default/
cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai2_id2,Z8,4,1,2_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig1/default/
cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai0_id2,Z1,3,1,1_class0_gt0_pred0_mean.jpg results/cnn_hs/paper_plots/fig1/default/
cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai0_id2,Z2,1,0,2_class0_gt0_pred0_mean.jpg results/cnn_hs/paper_plots/fig1/default/

cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai12_id2,Z16,2,1,1_class1_gt1_pred1.jpg results/cnn_hs/paper_plots/figures_wavelength_explanations/default/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai12_id2,Z16,2,1,1_class1_gt1_pred1.jpg results/cnn_hs/paper_plots/figures_wavelength_explanations/xil/

cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai0_id1,Z1,1,0,1_class0_gt0_pred0_mean.jpg results/cnn_hs/paper_plots/fig3/
cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai0_id1,Z2,2,1,0_class0_gt0_pred0_mean.jpg results/cnn_hs/paper_plots/fig3/
cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai0_id5,Z3,4,1,0_class0_gt0_pred0_mean.jpg results/cnn_hs/paper_plots/fig3/
cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai0_id2,Z1,4,1,0_class0_gt0_pred0_mean.jpg results/cnn_hs/paper_plots/fig3/
cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai6_id1,Z13,1,1,0_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig3/
cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai6_id1,Z9,2,1,2_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig3/
cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai12_id2,Z17,4,1,1_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig3/
cp results/model_checkpoints/cnn_hs/cv/default_4x13x13_cv1/default/gradcam_model_best/eval/plots/dai13_id3,Z14,3,0,0_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig3/

cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai0_id1,Z1,1,0,1_class0_gt0_pred0_mean.jpg results/cnn_hs/paper_plots/fig4/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai0_id1,Z2,2,1,0_class0_gt0_pred0_mean.jpg results/cnn_hs/paper_plots/fig4/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai0_id5,Z3,4,1,0_class0_gt0_pred0_mean.jpg results/cnn_hs/paper_plots/fig4/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai0_id2,Z1,4,1,0_class0_gt0_pred0_mean.jpg results/cnn_hs/paper_plots/fig4/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai6_id1,Z13,1,1,0_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig4/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai6_id1,Z9,2,1,2_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig4/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai12_id2,Z17,4,1,1_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig4/
cp results/model_checkpoints/cnn_hs/cv/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/plots/dai13_id3,Z14,3,0,0_class1_gt1_pred1_mean.jpg results/cnn_hs/paper_plots/fig4/

cp results/model_checkpoints/cnn_hs/cv_trained/xil_4x13x13_cv1/default/gradcam_checkpoint/eval/npy/* results/cnn_hs/gradcams_npy/xil
cp results/model_checkpoints/cnn_hs/cv_trained/default_4x13x13_cv1/default/gradcam_model_best/eval/npy/* results/cnn_hs/gradcams_npy/default