from pathlib import Path
from types import NoneType

import torch

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from ..CustomBatchSampler import OriginBatchSampler, \
    customBatchSampler_create_origin_indices
from ..rrr_dataset import RRRDataset
from ..utils import XILUtils
from .cnn import CNNTwoConv
import itertools
from ..caipi import RandomStrategy, SubstitutionStrategy, AlternativeValueStrategy, MarginalizedSubstitutionStrategy, \
    to_counter_examples_2d_pic, PseudoStrategy

from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import argparse
import progressbar
import sys
from torch.utils.tensorboard import SummaryWriter
from .caipi_grid_search import (
    save_info_to_csv, has_significant_improvement,
    define_optim, merge_config_with_args, ConfigManager)

import math
import torchvision.utils as vutils
from sklearn.metrics import accuracy_score, cohen_kappa_score

widgets = [
    progressbar.Percentage(),
    " ",
    progressbar.GranularBar(),
    " ",
    progressbar.Timer(),
]
sys.stdout = progressbar.streams.wrap_stdout()

image_shape = torch.Size((1, 1, 28, 28))
device = XILUtils.define_device()
batch_size = 64
num_classes = 2
label_translation = dict(zero=torch.tensor((1, 0), device=device), eight=torch.tensor((0, 1), device=device))
# TODO remove
classifier_target_output = ClassifierOutputTarget(1)
# TODO
softmax_func = torch.nn.Softmax(dim=-1)

standard_aggregation = lambda input_tensor: torch.argmax(input_tensor, dim=-1)

def train_loop(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn, optimizer, device: str, batch_size, original_data_size):
    size = len(dataloader.dataset)
    total_loss = 0

    interval = size // 10
    model.train()
    for batch_i, (X, y) in enumerate(dataloader):
        # move X and y to device
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        # get counter_example_ids
        counterexample_ids = [idx - original_data_size for idx in dataloader.batch_sampler.batch_second_idxs]
        is_counter_examples_mask = torch.zeros(len(X), dtype=torch.bool)
        is_counter_examples_mask[len(X) - k:] = 1
        loss = loss_fn(pred, y, is_counter_examples_mask, counterexample_ids)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if (batch_i * batch_size) % interval < batch_size:
            loss, current = loss.item(), batch_i * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            print(f"loss_fn: {loss_fn.__str__()}")

    return total_loss / len(dataloader)


class HingeLoss:
    def __init__(self, margin=0.0, reduction=torch.sum):
        self.margin = margin
        # self.w_softmax = w_softmax
        self.reduction = reduction

    def __call__(self, logits, targets,  *args, **kwargs):
        # # Prestep 1: Softmax
        # if self.w_softmax:
        #     logits = torch.nn.functional.softmax(logits, dim=-1)
        # Step 1: calculate loss
        loss = torch.clamp(self.margin - logits[:, targets.argmax(dim=-1)], min=0.0)
        # Step 2: Reduce to scalar tensor
        if self.reduction:
            loss = self.reduction(loss)
        return loss


class LagrangianLoss:
    def __init__(self, base_loss, model, tau, lambda_update_constant, lambda_update_interval, counterexamples, ce_labels,
                 evaluation=False, initial_lambda=0.1, fixed_lambda=False):
        super().__init__()
        self.base_loss = base_loss
        self.threshold = tau
        self.hinge_loss = HingeLoss(margin=tau, reduction=None)

        self.lambda_update_constant = lambda_update_constant
        self.lambda_update_interval = lambda_update_interval
        self.lambdas = {}  # will store lambda for each counterexample
        self.initial_lambda = initial_lambda

        self.step_count = 0
        self.evaluation = evaluation
        self.fixed_lambda = fixed_lambda

        self.counterexamples = counterexamples
        if not ce_labels is None:
            self.ce_labels = ce_labels.argmax(dim=-1)
        self.model = model

        if not ce_labels is None:
            self._init_lambda_dict()

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, is_counterexample_mask: torch.Tensor=None, counterexample_ids=None):
        # PreStep 1: Prepare Data
        if isinstance(is_counterexample_mask, NoneType):
            is_counterexample_mask = torch.zeros(len(predictions), dtype=torch.bool)
        original_predictions = predictions[~is_counterexample_mask]
        original_targets = targets[~is_counterexample_mask]

        ce_predictions = predictions[is_counterexample_mask]
        ce_targets = targets[is_counterexample_mask]

        # Step 1: Compute Loss for Original Data
        original_data_loss = self.base_loss(original_predictions, original_targets)

        # Step 2: Compute Loss for CounterExamples
        ce_loss = 0
        if not isinstance(counterexample_ids, NoneType):
            # Step 2.1: Check if Lambdas Need to be Updated
            if not self.evaluation:
                # TODO: maybe we should count steps for each lambda individually and only if it is used in calculation
                self.step_count += 1
                if self.step_count % self.lambda_update_interval == 0 and not self.fixed_lambda:
                    self._update_lambdas()
            # Step 2.2: Compute hinge loss
            hinge_loss = self.hinge_loss(ce_predictions, ce_targets)
            # Step 2.1: Multiply by Lambdas
            ce_loss = hinge_loss * self._get_lambdas(counterexample_ids)

        # Step 3: Reduce to Scalar Tensor
        out = original_data_loss + ce_loss
        return out

    def _init_lambda_dict(self):
        for i, ce in enumerate(self.counterexamples):
            self.lambdas[i] = self.initial_lambda

    def _update_lambdas(self):
        logits = self.model(self.counterexamples)
        logits = torch.nn.functional.softmax(logits, dim=-1)
        for i, logit in enumerate(logits):
            if logit[self.ce_labels[i]].item() >= threshold:
                # decrease lambda
                self.lambdas[i] /= self.lambda_update_constant
            else:
                # increase lambda
                self.lambdas[i] *= self.lambda_update_constant

    def _get_lambdas(self, counterexample_ids):
        # Step 1: Get Lambdas
        agg_arr = []
        for i in counterexample_ids:
            agg_arr.append(self.lambdas[i])
        # Step 2: Return Tensor of Lambdas
        return torch.tensor(agg_arr, device=self.counterexamples.device)

    def train(self):
        self.evaluation = False

    def evaluate(self):
        self.evaluation = True

    def __str__(self):
        # print info and all current lambdas with step number
        return (f"LagrangianLoss(step={self.step_count}, "
                f"lambda_update_constant={self.lambda_update_constant}, "
                f"lambda_update_interval={self.lambda_update_interval}, "
                f"initial_lambda={self.initial_lambda}, "
                f"threshold={self.threshold})\n"
                f"Lambdas: {self.lambdas}")


def define_parameters(inputs, targets, config_manager: ConfigManager, specific_case: str = None):
    """Define parameters using config manager with priority system."""

    # Initialize strategies
    random_strategy = RandomStrategy(0., 1., torch.float32)
    substitution_strategy = SubstitutionStrategy(inputs, targets)
    marginalized_substitution_strategy = MarginalizedSubstitutionStrategy(inputs, targets)
    alternative_value_strategy = AlternativeValueStrategy(torch.zeros(image_shape, device=device), image_shape)
    pseudo_strategy = PseudoStrategy()

    # Strategy mapping for config file
    strategy_mapping = {
        'random': random_strategy,
        'substitution': substitution_strategy,
        'marginalized_substitution': marginalized_substitution_strategy,
        'alternative_value': alternative_value_strategy,
        'pseudo': pseudo_strategy,
    }

    # Get parameters from config with fallback to defaults
    try:
        ce_num = config_manager.get_parameter('ce_num', specific_case, [2])
        strategy_names = config_manager.get_parameter('strategy', specific_case,
                                                      ['substitution', 'marginalized_substitution',
                                                       'alternative_value'])
        num_of_instances = config_manager.get_parameter('num_of_instances', specific_case, [3])
        lr = config_manager.get_parameter('lr', specific_case, [1e-1, 1e-2, 1e-3])
        lambda_update_constant = config_manager.get_parameter('lambda_update_constant', specific_case, math.sqrt(2))
        lambda_update_interval = config_manager.get_parameter('lambda_update_interval', specific_case, 5)
        initial_lambda = config_manager.get_parameter('initial_lambda', specific_case, 0.1)

        # Convert strategy names to strategy objects
        strategies = []
        for strategy_name in strategy_names:
            if strategy_name in strategy_mapping:
                strategies.append(strategy_mapping[strategy_name])
            else:
                available_strategies = list(strategy_mapping.keys())
                raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: {available_strategies}")

        parameters_grid = {
            "ce_num": ce_num if isinstance(ce_num, list) else [ce_num],
            "strategy": strategies,
            "num_of_instances": num_of_instances if isinstance(num_of_instances, list) else [num_of_instances],
            "lr": lr if isinstance(lr, list) else [lr],
            "lambda_update_constant": lambda_update_constant if isinstance(lambda_update_constant, list) else [lambda_update_constant],
            "lambda_update_interval": lambda_update_interval if isinstance(lambda_update_interval, list) else [
                lambda_update_interval],
            "initial_lambda": initial_lambda if isinstance(initial_lambda, list) else [initial_lambda]
        }

        print(f"Loaded parameters from config:")
        print(f"  ce_num: {parameters_grid['ce_num']}")
        print(f"  strategy: {[str(s) for s in parameters_grid['strategy']]}")
        print(f"  num_of_instances: {parameters_grid['num_of_instances']}")
        print(f"  lr: {parameters_grid['lr']}")
        print(f"  lambda_update_constant: {parameters_grid['lambda_update_constant']}")
        print(f"  lambda_update_interval: {parameters_grid['lambda_update_interval']}")
        print(f"  initial_lambda: {parameters_grid['initial_lambda']}")
        print()

        return parameters_grid

    except Exception as e:
        print(f"Error loading parameters from config: {e}")
        print("Falling back to hardcoded defaults...")

        # Fallback to original hardcoded parameters
        parameters_grid = {
            "ce_num": [2],
            "strategy": [substitution_strategy, marginalized_substitution_strategy, alternative_value_strategy],
            "num_of_instances": [3],
            "lr": [1e-1, 1e-2, 1e-3],
            "lambda_update_constant": [math.sqrt(2)],
            "lambda_update_interval": [5],
            "initial_lambda": [0.1]
        }
        return parameters_grid

def write_lambdas(writer: SummaryWriter, lambdas: dict, epoch: int, counterexamples_predictions):
    for (key, value), lambda_accuracy in zip(lambdas.items(), counterexamples_predictions.tolist()):
        writer.add_scalar(f"Lambda value/{key}", value, epoch)
        writer.add_scalar(f"Lambda accuracy/{key}", lambda_accuracy, epoch)


def fit_until_optimum_or_threshold(model, train_dataloader, test_dataloader, optimizer, loss_fn, original_data_size,
                                   writer,
                                   threshold, no_improve_epochs_th=10, evaluate_every_nth_epoch=10, train_dl_for_evaluation=None):
    epoch = 0
    epochs_no_improve = 0
    best_loss = float('inf')

    while True:
        # Step 1: Train
        loss_fn.train()
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device, batch_size, original_data_size)
        loss_fn.evaluate()
        metrics, avg_loss = XILUtils.test_loop(train_dl_for_evaluation, model, loss_fn, device, metric='accuracy')
        train_accuracy = metrics["accuracy"]
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Average Loss/train", train_loss, epoch)

        # Get Counterexamples predictions for accuracy calculation
        counterexamples_logits = model(loss_fn.counterexamples)
        counterexamples_predictions = softmax_func(counterexamples_logits)[:, 1] # explicitly choosing only confidences in eights
        # TODO: not really good for generalised solution

        write_lambdas(writer, loss_fn.lambdas, epoch,
                      counterexamples_predictions=counterexamples_predictions)

        # Step 2: Evaluate Now?
        if epoch % evaluate_every_nth_epoch == 0:
            # SubStep 1: Get Metrics data
            loss_fn.evaluate()
            metrics, avg_loss = XILUtils.test_loop(test_dataloader, model, loss_fn, device, metric='accuracy')
            acc = metrics["accuracy"]
            writer.add_scalar("Accuracy/val", acc, epoch)
            writer.add_scalar("Average Loss/val", avg_loss, epoch)

            # SubStep 2: If We Reach Set Threshold -> We Fitted and Exit
            # TODO: we temporarily disable exit on reaching threshold
            # if acc >= threshold:
            #     print(f"Reached accuracy threshold of {threshold:.2f} at epoch {epoch}.")
            #     break

        # Step 3: check if loss has significantly improved
        if has_significant_improvement(best_loss, train_loss, relative_eps=1e-2):
            best_loss = train_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= no_improve_epochs_th:
                print(f"Stopping training after {epochs_no_improve} epochs without improvement.")
                break
        epoch += 1

    # Report to Tensorboard
    final_lambdas = {f"Lambda {str(key)}": value for key, value in loss_fn.lambdas.items()}
    lambda_accuracies = {f"Lambda accuracy {str(key)}": value.cpu().item() for key, value in zip(list(loss_fn.lambdas.keys()), counterexamples_predictions)}

    metrics_dict = {"test_accuracy": acc, "train_accuracy": train_accuracy,
                   "test_average_loss": avg_loss, "train_average_loss": train_loss,"num_of_artificial_instances": len(loss_fn.lambdas),
                    **final_lambdas, **lambda_accuracies}
    return metrics_dict, avg_loss, writer

def grid_search(filename: Path, misleading_ds_train, model_confounded, test_dataloader, device, threshold, optim,
                fixed_lambda=False,
                evaluate_every_nth_epoch=16, from_ground_zero=False, config_manager: ConfigManager = None, specific_case: str = None):
    parameters_grid = define_parameters(misleading_ds_train.data.to(device), misleading_ds_train.labels.to(device),
                                      config_manager or ConfigManager(), specific_case)
    combinations = list(itertools.product(*parameters_grid.values()))

    misleading_data = misleading_ds_train.data.to(device)
    misleading_labels = misleading_ds_train.labels.to(device)
    misleading_binary_masks = misleading_ds_train.binary_masks.to(device)

    log_dir = f"runs/caipi_lagrange_experiment_{specific_case}"
    writer = SummaryWriter(
        log_dir=log_dir)
    writer_global_step = [1]
    records = list()
    for ce_num, strategy, num_of_instances, lr, lambda_update_constant, lambda_update_interval, initial_lambda in combinations:
        print(f"Checking out {ce_num=}, {strategy=}, {lr=}, {num_of_instances=}, {lambda_update_constant=}, {lambda_update_interval=},"
              f"{initial_lambda=}.")
        grid_search_iteration(ce_num, device, filename, from_ground_zero, lr, misleading_binary_masks,
                              misleading_data, misleading_labels, model_confounded, num_of_instances, optim,
                              evaluate_every_nth_epoch, strategy, test_dataloader, threshold, records, specific_case,
                              lambda_update_constant, lambda_update_interval, initial_lambda, fixed_lambda, writer, writer_global_step)
    # After Grid Search
    writer.close()


def define_langarngian_loss(model, counter_examples, ce_labels, threshold, lambda_update_constant=math.sqrt(2), lambda_update_interval=5,
                            evaluation=False, initial_lambda=0.1, fixed_lambda=False):
    return LagrangianLoss(torch.nn.CrossEntropyLoss(), model, threshold,
                          lambda_update_constant=lambda_update_constant, lambda_update_interval=lambda_update_interval,
                          counterexamples=counter_examples, ce_labels=ce_labels, evaluation=evaluation,
                          initial_lambda=initial_lambda, fixed_lambda=fixed_lambda)


def grid_search_iteration(ce_num, device, filename, from_ground_zero, lr, misleading_binary_masks,
                          misleading_data, misleading_labels, model_confounded, num_of_instances, optim,
                          evaluate_every_nth_epoch, strategy, test_dataloader, threshold, records, specific_case,
                          lambda_update_constant, lambda_update_interval, initial_lambda, fixed_lambda, writer, writer_global_step):
    original_data_size = misleading_data.size(0)
    used_indices = set()
    def get_informative_instance(targets, num_of_instances, dataset_size):
        # just take some random eight
        eight = label_translation["eight"]
        indices: torch.Tensor = torch.arange(dataset_size, device=device)[(targets == eight).all(dim=1)]
        # remove used indices
        indices = torch.tensor(list(set(indices.tolist()) - used_indices), device=device)

        # take a random eight
        perm = torch.randperm(len(indices), device=device)
        indices = indices[perm[:num_of_instances]]

        # update used_indices
        used_indices.update(indices.tolist())

        return indices



    grid_model = CNNTwoConv(num_classes, device)
    model_confounded_state_dict = model_confounded.state_dict()
    grid_model.load_state_dict(model_confounded_state_dict)

    optimizer = define_optim(optim, grid_model, lr)

    loss = define_langarngian_loss(grid_model, None, None,
                                   threshold=threshold,
                                   lambda_update_interval=lambda_update_interval,
                                   lambda_update_constant=lambda_update_constant,
                                   evaluation=True,
                                   initial_lambda=initial_lambda,
                                   fixed_lambda=fixed_lambda
                                   )
    metrics_dict, avg_loss = XILUtils.test_loop(test_dataloader, grid_model, loss, device, metric='all')
    accuracy = metrics_dict["accuracy"]
    print(f"Initial accuracy: {100 * accuracy:.2f}%")

    current_data = misleading_data.clone()
    current_labels = misleading_labels.clone()
    current_binary_masks = misleading_binary_masks.clone()

    counterexamples_epoch = 1
    duplicate_informative_instances_iteration = 0

    target_layers = [grid_model[3]]

    with progressbar.ProgressBar(widgets=widgets, max_value=1e+4) as bar:
        # Update ProgressBar
        bar.update(accuracy * 1e+4)
        # Step Into Fitting Until Convergence
        while accuracy < threshold:
            if from_ground_zero:
                grid_model.load_state_dict(model_confounded_state_dict)
                optimizer = define_optim(optim, grid_model, lr)

            # Step 1: Get Informative Instances
            indices = get_informative_instance(current_labels[:original_data_size], num_of_instances,
                                               original_data_size)
            if len(indices) == 0:
                print(f"\nAll original instances were used"
                      f" {duplicate_informative_instances_iteration + 1} times for creation of counterexamples."
                      f" New iteration...\n")
                duplicate_informative_instances_iteration += 1
                used_indices.clear()
                continue
            informative_instances = current_data[indices]

            # Step 3: Get Informative Targets and Corresponding Binary Masks
            informative_targets = current_labels[indices]
            informative_binary_masks = current_binary_masks[indices]

            # Step 4: Query Explanation
            explanation = informative_binary_masks.bool() # ! Assumption: we've got ideal explainer/annotator

            # Step 5: Generate CounterExamples
            counterexamples = to_counter_examples_2d_pic(strategy, informative_instances, explanation, ce_num,
                                                         target=label_translation["eight"].unsqueeze(0))
            bs, ce, ch, he, we = counterexamples.shape
            counterexamples = counterexamples.view(bs * ce, ch, he, we)

            # Step 6: Populate Dataset With New CounterExamples
            current_data = torch.vstack((current_data, counterexamples))
            current_labels = torch.vstack((current_labels, (informative_targets.repeat_interleave(ce_num, dim=0))))
            current_binary_masks = torch.vstack(
                (current_binary_masks, informative_binary_masks.repeat_interleave(ce_num, dim=0)))

            # Step additional: redefine loss to cotnain new counterexamples
            loss = define_langarngian_loss(grid_model, counter_examples=current_data[original_data_size:],
                                           ce_labels=current_labels[original_data_size:],
                                           threshold=threshold,
                                           lambda_update_constant=lambda_update_constant,
                                           lambda_update_interval=lambda_update_interval,
                                           evaluation=True, # it will be changed in `fit_until_optimum_or_threshold`
                                           initial_lambda=initial_lambda,
                                           fixed_lambda=fixed_lambda
                                           )

            num_of_artifical_instances = len(current_labels) - original_data_size
            # Step 7: Fit to new dataset
            current_tensor_dataset = TensorDataset(current_data, current_labels)
            first_origin_idxs, second_origin_idxs = customBatchSampler_create_origin_indices(original_data_size, len(current_tensor_dataset))
            batch_sampler = OriginBatchSampler(first_origin_idxs, second_origin_idxs, batch_size, k)
            grid_train_dl = DataLoader(current_tensor_dataset, batch_sampler=batch_sampler)
            grid_train_dl_normal = DataLoader(current_tensor_dataset, batch_size=batch_size, shuffle=False)

            # Fit until optimum
            metrics_dict, avg_loss, writer = fit_until_optimum_or_threshold(grid_model, grid_train_dl, test_dataloader, optimizer, loss, original_data_size,
                                                                    writer,
                                                                    threshold=threshold,
                                                                    evaluate_every_nth_epoch=evaluate_every_nth_epoch,
                                                                            train_dl_for_evaluation=grid_train_dl_normal)

            accuracy = metrics_dict["test_accuracy"]
            writer.add_hparams(
                {"ce_num": ce_num, "strategy": str(strategy), "lr": lr, "num_of_instances_per_epoch": num_of_instances,
                 "lambda_update_constant": lambda_update_constant,
                 "lambda_update_interval": lambda_update_interval, "initial_lambda": initial_lambda,
                 "threshold": threshold},
                metrics_dict,
                global_step=writer_global_step[0]
            )
            writer_global_step[0] += 1
            torch.save(grid_model.state_dict(), f"{writer.log_dir}/model_weights.pth")

            # Update ProgressBar
            bar.update(accuracy * 1e+4)

            # Update CSV report
            records.append({
                "ce_num": ce_num,
                "strategy": str(strategy),
                "lr": lr,
                "num_of_instances_per_epoch": num_of_instances,
                "lambda_update_constant": lambda_update_constant,
                "lambda_update_interval": lambda_update_interval,
                "initial_lambda": initial_lambda,
                "threshold": threshold,

                "counterexamples_iteration": counterexamples_epoch,

                "from_ground_zero": from_ground_zero,
                "num_of_artificial_instances": num_of_artifical_instances,
                **metrics_dict

            })
            print(f"Epoch {counterexamples_epoch}: Accuracy: {100 * accuracy:.2f}%, Avg. Test Loss: {avg_loss:.4f}")

            if num_of_artifical_instances > original_data_size // 2:
                # TODO: change only after the analysis of the results and if it shows that some model could converge
                break

            # update epoch
            counterexamples_epoch += 1
            break # TODO: we're forcing break temporarily, because we're not interested in more counterexamples for now
    df = pd.DataFrame(records)
    save_info_to_csv(filename, df)

    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run grid search with counterexample injection and save results to CSV",
        epilog = """
    Usage Examples:

      Basic usage without config:
        %(prog)s --threshold 0.95 --optimizer adam

      Use config file with general parameters:
        %(prog)s --config config.yaml

      Use specific configuration case:
        %(prog)s --config config.yaml --config_case quick_test

      Override config parameters with command line (highest priority):
        %(prog)s --config config.yaml --config_case comprehensive --threshold 0.99 --optimizer sgd

      Quick test run:
        %(prog)s --config config.yaml --config_case quick_test --train_dataset_size 100

      Full comprehensive search:
        %(prog)s --config config.yaml --config_case comprehensive --train_dataset_size -1

      Custom output location:
        %(prog)s --config config.yaml --output_filename experiments/my_experiment.csv

    Priority System (highest to lowest):
      1. Command line arguments
      2. Config file specific parameters (--config_case)
      3. Config file general parameters
      4. Default values

    Config File Structure:
      general:          # Default parameters for all experiments
        program_args:   # Program arguments (threshold, optimizer, etc.)
        ce_num: [...]   # Grid search hyperparameters
        strategy: [...]
        ...
      specific:         # Named experiment configurations
        quick_test:     # Specific case name
          program_args: # Override program arguments
          ce_num: [...] # Override hyperparameters
          ...
            """,
    formatter_class = argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Accuracy threshold at which to stop each run"
    )
    parser.add_argument(
        "--output_filename", "-o",
        type=Path,
        default=None,
        help="Path to the CSV file where results will be written"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adam", "sgd"],
        help="Optimizer to use for training the model"
    )
    parser.add_argument(
        "--train_from_ground_zero",
        action="store_true",
        help="If set, train from ground zero instead of using pretrained model with counterexmples from previous epochs",
        default=None
    )
    parser.add_argument(
        "--evaluate_every_nth_epoch",
        type=int,
        default=None,
        help="Number of epochs without improvement before stopping training"
    )
    parser.add_argument(
        "--train_dataset_size",
        type=int,
        default=None,
        help="Size of train dataset, Default value set to -1 corresponding to the whole dataset"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to configuration file (YAML or JSON format)"
    )
    parser.add_argument(
        "--config_case",
        type=str,
        default=None,
        help="Specific configuration case to use from config file"
    )
    parser.add_argument(
        "--fixed_lambda",
        type=bool,
        default=None,
        help="Whether lagrangian loss should use fixed lambda, which are equal to default value"
    )

    args = parser.parse_args()

    # Initialize config manager
    config_manager = ConfigManager(args.config, args.config_case) if args.config else ConfigManager()

    # Merge config with args (args have priority)
    final_args = merge_config_with_args(args, config_manager)

    # Print priority system warnings
    config_manager.print_priority_warnings(final_args)

    current_directory = Path(__file__).parent

    # Load the dataset
    dataset = torch.load(current_directory / "data/08MNIST/confounded_v1/train.pth", weights_only=False) # TensorDataset
    ds_size = final_args.train_dataset_size
    assert ds_size < len(dataset.tensors[0]), f"Dataset size: {ds_size} is set bigger than the actual dataset size."

    # shuffle indices
    indices = torch.randperm(len(dataset.tensors[0]))

    misleading_ds_train = RRRDataset(
        dataset.tensors[0][indices][:ds_size],
        dataset.tensors[1][indices][:ds_size],
        dataset.tensors[2][indices][:ds_size],
    )

    model_confounded = CNNTwoConv(num_classes, device)
    model_confounded.load_state_dict(
        torch.load(current_directory / "08_MNIST_output/model_confounded.pth", weights_only=True)
    )
    model_confounded = model_confounded.to(device)

    ds_test = torch.load(current_directory / "data/08MNIST/original/test.pth", weights_only=False) # TensorDataset
    test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    threshold = final_args.threshold
    output_file = final_args.output_filename\
        if isinstance(final_args.output_filename, Path) else Path(final_args.output_filename)
    optimizer = final_args.optimizer

    # TODO: replace hardcoded part
    k = 1
    # TODO

    fixed_lambda = final_args.fixed_lambda

    grid_search(
        output_file,
        misleading_ds_train,
        model_confounded,
        test_dataloader,
        device,
        threshold,
        optimizer,
        fixed_lambda=fixed_lambda,
        evaluate_every_nth_epoch=final_args.evaluate_every_nth_epoch,
        from_ground_zero=final_args.train_from_ground_zero,
        config_manager=config_manager,
        specific_case=final_args.config_case,
    )