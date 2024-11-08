# Description of workflow on 1st semester of bakalarka
## TODO
- [ ] Theory:
    - [ ] Find where RRR is applied
    - [ ] Research how it is done, what modules are used, on what kind of dataset it is done
    - [ ] Try to experiment with input parameters
    - [ ] Understand how can I evaluate model learning, where learning is done in short time to show speed of progress, and based on input parameters

- [ ] Building project:
    - [ ] Find EKG datasets
    - [ ] Try to build my model using RRR gradient

- [ ] Environment:
    - [ ] Research how to build singularity container in order for it to be run in RCI cluster
    - [ ] Build container with dependencies.

- [ ] RCI cluster:
    - [ ] Research how to schedule and use rci cluster, maybe ask Professor or admins of RCI for manual.

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

## QAs
- [ ] 

## Flow
### Find where RRR is applied
<!-- TODO: DELETE ❌✅ -->
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
<!-- * Plant_Phenotyping -->
