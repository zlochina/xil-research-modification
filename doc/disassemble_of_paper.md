# Disassembling XIL paper
<!-- - TODO: What is CAIPI exactly? -->
## QAs
- How the authors of the paper catch the instances, where the case of "Right for the wrong reasons" happened? Do they check every explanation in order to check if its 
## Ideas of XIL
- destroy "Clever Hans"-like behavior
- provide a tool for creating trus

## Other general stuff
- "Clever Hans"-like behavior - making use of confounding factors within datasets

- LIME - is a method of *local explainer*. decision surface of the classifier (maybe other models too) can be locally approximated by a simple. interpretable local model.
    - the local model is defined in terms of simple features (encoding the presence/absence of basic components e.g. objects in a picture). An explanation can be readily extracted by reading off the contributions of the various components to the target prediction. **My note**: not sure how it works.
- local explainer - stategy for explaining machine learning models focusing on the task of explaining individual predictions.

- Active learning - such a mechanism, where model does not have labels to every instance provided, instead it only has a very limited set of examples. When the model is unsure or reaches some other criterium it can query the user/information source for a label to a given instance for improving itself/refitting itself.
- Example is a combination of instance and label
    - Instance is x
    - Label is y
    - y = f(x), where f is the model.

## Intro
- An example of the scientist, which trained DNN (deep NN) to recognise plant pathogens, which resulted in high accuracy, however when applied explainable AI, the confounding factors are found to influence the results.
- Proclaiming an idea, that User can correct the model's problem "Right predictions for the wrong reasons" towars making "the right predictions for the right reasons", which in its order could increase trust towards machine learning techniques.
- "The link between interacting, explaining and building trust has been largely ignored by the ML literature". Existing approaches focus on passive learning only, interactive learning frameworks such as active and coactive learning do not consider the issue of trust
- **Novel** "Explanatory interactive learning" (XIL) stands for both explanatory learning (models are forced to show explanation for their beliefs) and interactive learning (User corrects the model to go in the right way).
- Novelty: "we add the scientist into the training loop, who interactively revises the original model by interacting via it's explanations so that the model produces trustworthy decisions without a major drop in perfomance."
- Interaction is as follows:
    - In each step, **the learner** (the model) explains its interactive query to **the domain expert** (User), and **the expert** responds by correcting the explanations, **if necessary**, thus providing feedback. This allows to check the rightfulness of the prediction and rightfulness of the reasons for the said prediction. Automatically this mechanism provides so-called "witnessing of the evolution of the explanations" - the human user can see whether the model eventually "gets it".
- What practical tasks were done within the paper:
    1) Introduction of XIL with **counterexamples (CE)** to revise "Clever Hans" behavior in a model-agnostic fashion (independent from the type of the model).
    2) Adaption of the **"right for the right reasons" (RRR)** loss to latent layers of deep neural networks <!-- TODO: What does it mean exactly? -->
    3) Showcasing XIL on the computer vision benchmark datasets PASCAL VOC 2007 and MSCOCO 2014
    4) Evaluation of XIL on a highly relevant dataset for plant phenotyping demonstrating its potential as an enabler of scientific discovery
    5) Gathering of the plant phenotyping dataset and the creation of a version with confounders.
    6) A User study on trust development within XIL. demonstrate the importance of explaining decisions for building trustful machines

## Explanatory Interactive Machine Learning (XIL)
### Points of interest
- **A learner** can interactively query the user (or some other information source) to obtain the desired outputs of the data points. My note: should we assume from this that it is said in the context of Unsupervised learning. The interaction is as follows:
    - At each step, **the learner** considers a data point (labeled or unlabeled), predicts a label, and provides explanations of its prediction.
    - **The user** responds by correcting **the learner** if necessary, providing a slightly improved (not necessarily optimal) feedback to the learner.
- Framework of Explanatory Active learning (which is used in paper):
    - **Active learner** provides:
        1) `SelectQuery(f, U)` - procedure for selecting an informative instance x ∈ *U* based on the current model _f_ <!-- TODO explain what it means -->
        2) `Fit(L)` - fitting new model/updating current model *f* on the examples of in *L*.
    - **Explainer** provides:
        1) `Explain(f, x, ý)` - procedure providing explanation for a particular prediction ý = f(x).
- TODO
- During interactions between **the learner** and **the expert** 3 scenarios could occur:
    1) **Right for the right reasons** - no feedback is requested
    2) **Wrong for the wrong reasons** - in active learning we request User to provide the correct label. The explanation may provide info on why the prediction was wrong, however this is not discussed in the scope of this paper - so **no action** done.
    3) **Right for the wrong reasons** - the prediction is correct, the explanation is wrong - main target of XIL.

## Model-agnostic XIL using counterexamples (CE)
- In the experiments of the paper the expert (simulated one) indicate the components that have been wrongly identified by the explanation as relevant. Notation: C = {j : |wj | > 0 ∧ the user believes the jth component to be irrelevant}
    - In simple words correction **C** is a set of data indicating what data/factors expert found to be irrelevant to the prediction.
- Explaining it back to **the learner**:
    - Strategy embodied by **ToCounterExamples**. It converts **C** to a set of counterexamples

## RRR using gradiens
\[
L(\theta, X, y, A) = \sum_{n=1}^{N} \sum_{k=1}^{K} -c_k y_{nk} \log (\hat{y}_{nk}) \quad \text{(Right answers)} 
\]

\[
+ \lambda_1 \sum_{n=1}^{N} \sum_{d=1}^{D} \left( A_{nd} \frac{\delta}{\delta h_{nd}} \sum_{k=1}^{K} c_k \log (\hat{y}_{nk}) \right)^2 \quad \text{(Right reasons)}
\]

\[
+ \lambda_2 \sum_{i} \theta_i^2 \quad \text{(Weight regularization)}
\]

This formula corresponds to the loss function incorporating three components:
1. **Right answers**: the standard cross-entropy loss.
2. **Right reasons**: a regularization term penalizing gradients that do not align with a binary mask \(A\), ensuring the model focuses on the correct features.
3. **Weight regularization**: regularization term to prevent overfitting by controlling the magnitude of weights.

Notation of N, K, D:
1. **n**: This represents the index of a data point in the dataset. **N** samples
2. **k**: This represents the class index or output dimension. **K** classes
3. **d** This represents the input feature dimension.
