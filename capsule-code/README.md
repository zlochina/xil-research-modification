# XIL
Repository for the Paper "Making deep neural networks right for the right scientific reasons by interacting with their explanations" by Patrick Schramowski, Wolfgang Stammer, Stefano Teso, Anna Brugger, Franziska Herbert, Xiaoting Shao, 
Hans-Georg Luigs, Anne-Katrin Mahlein & Kristian Kersting.

## Overview
This repository aims to reproduce the results of the above manuscipt.
Code for training neural networks and evaluating the resulting models on a toy dataset and on a real dataset is provided. 
Regarding the real dataset we also provide the trained model checkpoints.
Also the code for the statistical analysis of the user study "Trust development during Explanatory Interactive Learning" is provided.

## Content
* User Study - Statistical Analysis
* Plant Disease Detection (hyperspectral) - CNN Training, Evalutation, Generation of Explanations
* Plant Disease Detection (pseudo-RGB) - CNN Training, Evalutation, Generation of Explanations
* Cluster Plots of Explanation Strategies
* Toy Dataset Decoy Fashion MNIST - Training and Evalutation of CE, RRR and default


## Reproducing Paper Results

#### Before you start
**Computation time** of the "reproducible run" is **approx. 5h**.

By default we do not run the training process of the hyperspectral or RGB CNN in the "reproducible run". If it is desired to run the training and ignore the provided model checkpoint, please comment in following lines in the *code/run* file: 13 and 17 <br />
And comment out following lines: 12 and 16.<br />
This however, would take several days.

Therefore, **we provide the trained model checkpoints** with which one is able to reproduce the reported results, i.e. evaluate the trained model on the test set and generate the explanations for both setups -- default and XIL.

For **reproducing the training of the CNNs**, we recommend to run it on current GPU/TPU systems instead of the code ocean cloud workstation.
Training a CNN using the hyperspectral plant dataset took around 20h on 1 Nvidia Tesla V100 32GB. 
Please also note that launching the training code on code ocean requires also a smaller batchsize (b=6) than mentioned in the original experiments (b=10) because the GPU resources are limited.
This could result in diverged results compared to the reported results in the manuscript. Please consider to use the provided trained model checkpoints to reproduce the exact results reported in the manuscript.

### User Study
Statistical analysis are first computed in R. Plots are created in python. See lines 19 and 20 in run file.
Results (Figure 5) are saved in directory *user_study/* and are present in the output.

### Plant Disease Detection HS
By default we only generate the explanations of one fold (the one we used to plot the figures in the manuscript) of the cross validation. If it is desired to run all cross validation folds please comment in the required lines in the scripts located in *code/run_scripts/run_scripts_hs/[without_training or with_training]/run_gradcam_[default or xil]_cv.sh*.

The results are saved in *cnn_hs/evaluation/xil* and *default/evaluation_cvi.txt* (Balanced Accuracy Table 1) and *cnn_hs/paper_plots/...* (Figures 1, 3, 4, 6). In the manuscript we only show visual results from the fold 1. 
If you want to generate the explanations of the other cross-validation folds keep in mind that the resulting explanations are not copied to the results directory and are therefore removed after code oceans "reproducible run" finished. To copy the results please adapt the script *code/run_scripts/run_scripts_hs/without_training/copy_figure_explanations.sh*

### Plant Disease Detection RGB
By default we only generate the explanations of one fold (the one we used to plot the figures in the manuscript) of the cross validation. If it is desired to run all cross validation folds please comment in the required lines in the scripts located in *code/run_scripts/run_scripts_rgb/[without_training or with_training]/run_gen_cams_rgb_[default or rrr].sh*.

The accuracies are printed in the log file. The resulting grad cams displayed in the manuscript are stored in *cnn_rgb/paper_plots/[fig3 and fig4]*. All grad cams are stored in *model_checkpoints/cnn_rgb/cv/[default or rrr]_cv2/gradcams/* (Note: if a different cross validation is run the previous two grad cam paths will denote a different number for the *default_cv* and *rrr_cv*). In the manuscript we only show visual results from the fold 2.

### Plant Disease Detection Cluster Analysis of the Decision Strategies
Cluster Plots are saved in the directories *cnn_hs/paper_plots/fig[3-4]/strategy_analysis/* and *cnn_rgb/paper_plots/fig[3-4]/strategy_analysis/* and the selected cluster samples can be found in *cnn_hs/paper_plots/fig[3-4]/* and *cnn_rgb/paper_plots/fig[3-4]/*.

### Toy Dataset - Fashion MNIST
The methods Counterexamples and RRR are tested on the toy experiment Decoy Fashion MNIST.
Results can be found in ouput.

## Results

* User Study - Statistical Analysis:

    The detailed results are located in the output file: *output.txt*. <br/>
    The figure (Fig. 5) shown in the manuscript is generated and saved to *userstudy/user_study_boxplots.pdf*

* Plant Disease Detection (hyperspectral) - CNN Training, Evalutation, Generation of Explanations (cross validation fold 1):
    
    **Accuracy** (table 1):<br/>
    Default (without correction):      Total Acc@1 98.83 | Acc@1 Balanced 99.26<br/>
    Explanatory Interactive Learning:  Total Acc@1 98.16 | Acc@1 Balanced 98.84<br/><br/>
    For more details compare output located in *cnn_hs/evaluation/default/evalutation_cv1.txt* and *cnn_hs/evaluation/xil/evalutation_cv1.txt*
    
    **Explanations** (figure 1, 3, 4, 6):<br/>
    Generated sample explanations (used in the manuscripts figures) for both uncorrected and corrected model are located in the corrospending directory located in *cnn_hs/paper_plots/*.
    
* Plant Disease Detection (RGB) - CNN Training, Evalutation, Generation of Explanations:

    **Accuracy** (table 1):<br/>
    Default (without correction):      Test accuracy of cv run 2: 0.9293 <br/>
    Explanatory Interactive Learning:  Test accuracy of cv run 2: 0.9321 <br/><br/>

    **Explanations** (figure 3, 4):<br/>
    Generated sample explanations (used in the manuscripts figures) for both uncorrected and corrected model are located in the corresponding directory located in *cnn_rgb/paper_plots/*.

* Cluster Plots of Explanation Strategies:

    **Explanations** (figure 3, 4):<br/>
    Generated cluster plots (used in the manuscripts figures) for both uncorrected and corrected model are located in the corresponding directory located in *cnn_rgb/paper_plots/fig[3-4]/strategy_analysis/* and *cnn_hs/paper_plots/fig[3-4]/strategy_analysis/*.
    
* Toy Dataset Decoy Fashion MNIST - Training and Evalutation of CE, RRR and default:

    Results MLP corrected on 1 examples (CE):<br />
    avg. acc. on train (decoy)  0.9439<br />
    avg. acc. on test (decoy)   0.8352<br /><br />
    Results MLP corrected on 3 examples (CE):<br />
    avg. acc. on train (decoy)  0.9347<br />
    avg. acc. on test (decoy)   0.8603<br /><br />
    Results MLP corrected on 5 examples (CE):<br />
    avg. acc. on train (decoy)  0.9238<br />
    avg. acc. on test (decoy)   0.8547<br /><br />
    Results MLP annotated (RRR):<br />
    avg. acc. on train (decoy)  0.9039<br />
    avg. acc. on test (decoy)   0.8579<br /><br />
    Results MLP normal (default):<br />
    avg. acc. on train (decoy)  0.9616<br />
    avg. acc. on test (decoy)   0.4808<br />