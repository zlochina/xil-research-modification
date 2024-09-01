mkdir -p results/cnn_hs/evaluation/xil/
mkdir -p results/cnn_hs/evaluation/default/

cp data/model_checkpoints/cnn_hs/cv_trained/xil_4x13x13_cv0/default/evaluation.txt ./results/cnn_hs/evaluation/xil/evaluation_cv0.txt
cp data/model_checkpoints/cnn_hs/cv_trained/xil_4x13x13_cv1/default/evaluation.txt ./results/cnn_hs/evaluation/xil/evaluation_cv1.txt
cp data/model_checkpoints/cnn_hs/cv_trained/xil_4x13x13_cv2/default/evaluation.txt ./results/cnn_hs/evaluation/xil/evaluation_cv2.txt
cp data/model_checkpoints/cnn_hs/cv_trained/xil_4x13x13_cv3/default/evaluation.txt ./results/cnn_hs/evaluation/xil/evaluation_cv3.txt
cp data/model_checkpoints/cnn_hs/cv_trained/xil_4x13x13_cv4/default/evaluation.txt ./results/cnn_hs/evaluation/xil/evaluation_cv4.txt

cp data/model_checkpoints/cnn_hs/cv_trained/default_4x13x13_cv0/default/evaluationmodel_best.txt ./results/cnn_hs/evaluation/default/evaluation_cv0.txt
cp data/model_checkpoints/cnn_hs/cv_trained/default_4x13x13_cv1/default/evaluationmodel_best.txt ./results/cnn_hs/evaluation/default/evaluation_cv1.txt
cp data/model_checkpoints/cnn_hs/cv_trained/default_4x13x13_cv2/default/evaluationmodel_best.txt ./results/cnn_hs/evaluation/default/evaluation_cv2.txt
cp data/model_checkpoints/cnn_hs/cv_trained/default_4x13x13_cv3/default/evaluationmodel_best.txt ./results/cnn_hs/evaluation/default/evaluation_cv3.txt
cp data/model_checkpoints/cnn_hs/cv_trained/default_4x13x13_cv4/default/evaluationmodel_best.txt ./results/cnn_hs/evaluation/default/evaluation_cv4.txt
