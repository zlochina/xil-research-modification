# Description of workflow on 1st semester of bakalarka
## TODO
- [ ] Theory:
    - [x] Find where RRR is applied
    - [ ] Research how it is done, what modules are used, on what kind of dataset it is done:
        - [ ] We can use as a template project's implementation of MLP and even Tensorflow version of MLP (multilayer percpetron).
        - [ ] As we're gonna replace dataset to another. Find out how it could be done. I guess usage of MLP and particular datasets would be found in the same file in each of the experiment from `experiments/*.ipynb`
    - [ ] Try to experiment with input parameters
    - [ ] Understand how can I evaluate model learning, where learning is done in short time to show speed of progress, and based on input parameters
    - [ ] Explanation methods. I saw LinearExplnation and gradcam in the project.
    - [ ] Finish disassemble of the [paper](./flow.md)
    - [ ] Create a dictionary of the terminology (e.g. active learning, XIL, Grad-CAM, LIME, ...)

- [ ] Building project:
    - [ ] Find EKG datasets
    - [ ] Try to build my model using RRR gradient

- [ ] Environment:
    - [x] Research how to build singularity container in order for it to be run in RCI cluster
    - [ ] Build container with dependencies.
        - [x] OS dependency
        - [ ] Python dependency. TODO: Find suitable dependency. Is resolved by finding common version for every `pyproject.toml`
        - [x] Python requirements dependencies (I guess we should try Poetry or try installing to main Pip)
        - [ ] Optional dependencies
            - [ ] Ranger. Should find out how to use it properly
            - [x] Starship. 
            - [ ] btop
            - [ ] Shell command autocompletion
    - [x] Try out building container using Singularity
    - [ ] Build poetry files for **Fashion_MNIST** and **Plant Phenotyping**. Optionally for Strategy Analysis
        - [x] Fashion_MNIST
        - [ ] Plant Phenotyping


- [ ] RCI cluster:
    - [ ] [Research](https://login.rci.cvut.cz/wiki/how_to_start) how to schedule and use rci cluster, maybe ask Professor or admins of RCI for manual.

- [ ] Source:
    - [x] Research how to work with pytorch
    - [ ] Add binary mask A to fit function or to loss function
    - [ ] Research the code from Plant Phenotyping. (Usage of Dataset, )
    - [ ] How did they add binary masks to datasets.
    - [ ] Build jupyter notebook of MLP usage

## Tree of the paper code
- [ ] `code/`
- [ ] `├── Fashion_MNIST`
- [ ] `│   └── caipi`
- [ ] `│       ├── caipi`
- [ ] `│       └── rrr`
- [ ] `│           ├── bin`
- [ ] `│           ├── data`
- [ ] `│           ├── experiments`
- [ ] `│           └── rrr`
- [ ] `├── Plant_Phenotyping`
- [ ] `│   ├── hs_dataset_splits`
- [ ] `│   │   ├── cv0`
- [ ] `│   │   ├── cv1`
- [ ] `│   │   ├── cv2`
- [ ] `│   │   ├── cv3`
- [ ] `│   │   └── cv4`
- [ ] `│   ├── hs_utils`
- [ ] `│   ├── rgb_dataset_splits`
- [ ] `│   └── rgb_utils`
- [ ] `│       └── __pycache__`
- [ ] `├── Strategy_Analysis_Visualization`
- [ ] `│   └── libs`
- [ ] `│       └── auto_spectral_clustering`
- [ ] `│           ├── __pycache__`
- [ ] `│           └── fig`
- [ ] `├── User_Study`
- [ ] `└── run_scripts`
- [ ] `    ├── run_scripts_fashion`
- [ ] `    ├── run_scripts_hs`
- [ ] `    │   ├── with_training`
- [ ] `    │   └── without_training`
- [ ] `    ├── run_scripts_rgb`
- [ ] `    │   ├── with_training`
- [ ] `    │   └── without_training`
- [ ] `    ├── run_scripts_strategy_analysis`
- [ ] `    └── run_scripts_user_study`

## Summary. What was done?
- [ ] Experimented with containers to come up with the environment working for all subprojects to no avail. At least, we've got environment with CUDA and Poetry Python, which could be run on RCI cluster gpus.
- [ ] Gone through the guide of PyTorch. I guess I will try to build project on PyTorch, which lets running code on CUDA and gives interfaces and implementations for Dataset, Loss function, Model.
- [ ] ! I haven't been able to test their code as in the end I haven't come up with needed environment to run code adequately.

## What is planned to be done?
- [ ] Build Loss function of differential RRR using binary masks. In progress
- [ ] Build models using RRR loss function (where binary mask is optional thus RRR Loss function transforms to Cross Entropy Loss function). And apply it on the small datasets used by Research paper (in Fashion MNIST some datasets are automatically generated with the respective binary masks). Try it out in jupyter notebook.
- [ ] Dataset search. [LLM hint](https://www.perplexity.ai/search/for-my-project-i-need-ekg-data-X0uyEyZ.RnGNSoC80rWxbQ), also try out searching Hugging Face
- [ ] After finding datasets try to get acquainted with it. Understand whether binary masks could be generated.
- [ ] Still I haven't started on understanding GradCam or LinearExplanation methods.

## QAs
- [ ] Meaning of RRR? Does RRR in out project stands for the differential formula or the whole concept - actually, I mean if we should include discrete algorithms like CounterExamples? For now I included in my view only diffrential formula.
- [ ] EKG dataset:
    - [ ] Where could we get them? What kind of data it is (images, binary file,)
    - [ ] How do we create binary masks? Do we create for all of the examples? If not, we would need to see how the loss function behaves on rather simillar examples with and without "Right reasons" loss.
- [ ] Optimisation of hyperparameters. If you have some advices or guide, because I do not feel sure about this aspect.
- [ ] Model training optimizer. Do we use SGD or should we consider e.g. Adam etc. Im not really sure how it affects RRR loss function (as "Right reasons" part is actually using gradient)
- [ ] Gradient usage in "Right reasons"? Why do we apply binary mask to gradient, and not input parameters (I guess there is an answer somewhere in the materials of the paper.)
- [ ] Model architecture. Plant Phenotyping hyperspectral dataset CNN architecture was used. I cannot come up with potential architectures until I see data.
- [ ] Implementation of RRR loss function. Due to specifics of framework PyTorch, CrossEntropyLoss class has implemented L2 regularization, which I want to use it in such way that RRR loss will be built as sum of CrossEntropyLoss(with L2 reg) and manually defined Right Reasons loss. Im 90% sure, that there is no connection between Right Reasons loss and L2 regularization, is my assumption okay?
- [ ] Training is implemented by back propagation. That's alright isn't it

## Flow
### Find where RRR is applied
<!-- TODO: DELETE ❌✅ -->
Further `RRR` stands for `Right Reasons` part of the RRR loss function of the differentiable models
* Fashion_MNIST:
    * ✅caipi/caipi (definition of classes used in parent directory modules):
    * ✅caipi/rrr/bin:
    * ✅caipi/rrr/data:
    * ✅caipi/rrr/experiments (I guess its usage of rrr implementations):
        1. 2D Intuition.ipynb: usage of MultilayerPerceptron, experminet with input gradients
        2. 20 Newsgroups.ipynb: usage of MultilayerPerceptron, experiment with LIME
        3. Decoy MNIST.ipynb: usage of MultilayerPerceptron, experiment with LIME
        4. ...
    * ✅caipi/rrr/rrr:
        1. multilayer_perceptron.MultilayerPerceptron.objective: RRR implementation
        2. tensorflow_perceptron.TensorflowPerceptron.loss_function
    * ✅caipi:
        1. caipi.caipi(): CounterExamples. `problem.query_label(i);corrections.update()`
        2. versus-rrr (main flow): Usage of CE and RRR. Flow looks like comparisson of the algorithms
* Plant_Phenotyping:
    * ✅hs_utils:
        1. rrr_loss_hs.rrr_loss_function: RRR implementation
    * ✅rgb_utils:
        1. rrr_loss_rgb.rrr_loss_function: RRR implementation very simmilar to hs RRR implementation
    * ✅.:
        1. main_hs.train: Usage of rrr_loss_function from rrr_loss_hs.py
        2. main_rgb.train: Usage of rrr_loss_function from rrr_loss_rgb.py
