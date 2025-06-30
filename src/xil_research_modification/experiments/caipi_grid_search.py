from pathlib import Path
import torch
import yaml
import json
from typing import Dict, Any, Union
import warnings

from ..rrr_dataset import RRRDataset
from ..utils import XILUtils
from .cnn import CNNTwoConv
import itertools
from ..caipi import RandomStrategy, SubstitutionStrategy, AlternativeValueStrategy, MarginalizedSubstitutionStrategy, \
    to_counter_examples_2d_pic

from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import argparse
import progressbar
import sys
from torch.utils.tensorboard import SummaryWriter

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


class ConfigManager:
    """Manages configuration loading and parameter resolution with priority system."""

    def __init__(self, config_path: Path = None, config_case: str = None):
        self.config = {}
        self.config_case = config_case
        if config_path and config_path.exists():
            self.load_config(config_path)

    def load_config(self, config_path: Path):
        """Load configuration from YAML or JSON file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    self.config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    self.config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        except Exception as e:
            warnings.warn(f"Failed to load config file {config_path}: {e}")
            self.config = {}

    def get_parameter(self, param_name: str, specific_case: str = None, default_value: Any = None):
        """
        Get parameter value with priority system:
        1. Specific case parameters
        2. General parameters
        3. Default value
        4. Raise error if no value found and no default
        """
        # Check specific case first
        if specific_case and 'specific' in self.config:
            specific_config = self.config['specific'].get(specific_case, {})
            if param_name in specific_config:
                return specific_config[param_name]

        # Check general parameters
        if 'general' in self.config and param_name in self.config['general']:
            return self.config['general'][param_name]

        # Use default if provided
        if default_value is not None:
            return default_value

        # Raise error if no value found
        raise ValueError(f"Parameter '{param_name}' not found in config and no default provided")

    def get_program_args_from_config(self) -> Dict[str, Any]:
        """Extract program arguments from config."""
        program_args = {}

        # Get from general config
        if 'general' in self.config and 'program_args' in self.config['general']:
            program_args.update(self.config['general']['program_args'])

        # Get from specific configs (if any default specific case is defined)
        if 'specific' in self.config:
            if self.config_case in self.config['specific']:
                case_config = self.config['specific'][self.config_case]
                if 'program_args' in case_config:
                    program_args.update(case_config['program_args'])

        return program_args

    def print_priority_warnings(self, args: argparse.Namespace):
        """Print warnings about parameter priority system."""
        print("=" * 60)
        print("CONFIGURATION PRIORITY SYSTEM")
        print("=" * 60)
        print("Priority order (highest to lowest):")
        print("1. Command line arguments")
        print("2. Config file specific parameters")
        print("3. Config file general parameters")
        print("4. Default values")
        print()

        config_args = self.get_program_args_from_config()
        overridden_params = []

        for arg_name, arg_value in vars(args).items():
            if arg_name in config_args:
                config_value = config_args[arg_name]
                if arg_value != config_value:
                    overridden_params.append((arg_name, config_value, arg_value))

        if overridden_params:
            print("OVERRIDDEN PARAMETERS:")
            for param_name, config_val, arg_val in overridden_params:
                print(f"  {param_name}: config={config_val} -> args={arg_val}")
        else:
            print("No parameters overridden by command line arguments.")
        print("=" * 60)
        print()


def save_info_to_csv(filename: Path, info: pd.DataFrame):
    # Ensure the parent directory exists
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Write the info dictionary to the file as CSV
    info.to_csv(filename.with_suffix('.csv'), index=False)


def train_evaluate(model, train_dataloader, test_dataloader, optimizer, loss_fn, num_epochs=1, verbose=True):
    df = pd.DataFrame({
        "epoch": [],
        "accuracy": [],
        "cohen_kappa": [],
        "test_avg_loss": []
        })
    for epoch in range(num_epochs):
        # train loop
        XILUtils.train_loop(model, train_dataloader, optimizer, loss_fn, device)

        # Evaluate the model
        metrics_results, avg_test_loss = XILUtils.test_loop(test_dataloader, model, loss_fn, device, metric='all')
        new_data_row = {
            "epoch": epoch + 1,
            "accuracy": metrics_results["accuracy"],
            "cohen_kappa": metrics_results["kappa"],
            "test_avg_loss": avg_test_loss,
        }
        df.loc[len(df)] = new_data_row
    if verbose:
        print("Done training!")
    return df


def define_parameters(inputs, targets, config_manager: ConfigManager, specific_case: str = None):
    """Define parameters using config manager with priority system."""

    # Initialize strategies
    random_strategy = RandomStrategy(0., 1., torch.float32)
    substitution_strategy = SubstitutionStrategy(inputs, targets)
    marginalized_substitution_strategy = MarginalizedSubstitutionStrategy(inputs, targets)
    alternative_value_strategy = AlternativeValueStrategy(torch.zeros(image_shape, device=device), image_shape)

    # Strategy mapping for config file
    strategy_mapping = {
        'random': random_strategy,
        'substitution': substitution_strategy,
        'marginalized_substitution': marginalized_substitution_strategy,
        'alternative_value': alternative_value_strategy
    }

    # Get parameters from config with fallback to defaults
    try:
        ce_num = config_manager.get_parameter('ce_num', specific_case, [2])
        strategy_names = config_manager.get_parameter('strategy', specific_case,
                                                      ['substitution', 'marginalized_substitution',
                                                       'alternative_value'])
        num_of_instances = config_manager.get_parameter('num_of_instances', specific_case, [3])
        lr = config_manager.get_parameter('lr', specific_case, [1e-1, 1e-2, 1e-3])

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
        }

        print(f"Loaded parameters from config:")
        print(f"  ce_num: {parameters_grid['ce_num']}")
        print(f"  strategy: {[str(s) for s in parameters_grid['strategy']]}")
        print(f"  num_of_instances: {parameters_grid['num_of_instances']}")
        print(f"  lr: {parameters_grid['lr']}")
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
        }
        return parameters_grid

def has_significant_improvement(best_loss, current_loss, relative_eps=1e-3):
    """
    Checks whether current_loss is significantly better than best_loss
    based on a relative epsilon threshold.
    """
    threshold = best_loss * (1 - relative_eps)
    return current_loss < threshold

def fit_until_optimum_or_threshold(model, train_dataloader, test_dataloader, optimizer, loss_fn,
                                   threshold, no_improve_epochs_th=10, evaluate_every_nth_epoch=10):
    epoch = 0
    epochs_no_improve = 0
    best_loss = float('inf')

    while True:
        # Step 1: Train
        train_loss = XILUtils.train_loop(train_dataloader, model, loss_fn, optimizer, device)

        # Step 2: Evaluate Now?
        if epoch % evaluate_every_nth_epoch == 0:
            # SubStep 1: Get Metrics data
            metrics, avg_loss = XILUtils.test_loop(test_dataloader, model, loss_fn, device, metric='all')
            acc = metrics["accuracy"]

            # SubStep 2: If We Reach Set Threshold -> We Fitted and Exit
            if acc >= threshold:
                print(f"Reached accuracy threshold of {threshold:.2f} at epoch {epoch}.")
                break

        # Step 3: check if loss has significantly improved
        if has_significant_improvement(best_loss, train_loss, relative_eps=1e-3):
            best_loss = train_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= no_improve_epochs_th:
                print(f"Stopping training after {epochs_no_improve} epochs without improvement.")
                break
        epoch += 1
    return metrics, avg_loss

def grid_search(filename: Path, misleading_ds_train, model_confounded, test_dataloader, device, loss, threshold, optim,
                evaluate_every_nth_epoch=16, from_ground_zero=False, config_manager: ConfigManager = None, specific_case: str = None):
    parameters_grid = define_parameters(misleading_ds_train.data.to(device), misleading_ds_train.labels.to(device),
                                      config_manager or ConfigManager(), specific_case)
    combinations = list(itertools.product(*parameters_grid.values()))

    misleading_data = misleading_ds_train.data.to(device)
    misleading_labels = misleading_ds_train.labels.to(device)
    misleading_binary_masks = misleading_ds_train.binary_masks.to(device)



    records = list()
    for ce_num, strategy, num_of_instances, lr in combinations:
        grid_search_iteration(ce_num, device, filename, from_ground_zero, loss, lr, misleading_binary_masks,
                              misleading_data, misleading_labels, model_confounded, num_of_instances, optim,
                              evaluate_every_nth_epoch, strategy, test_dataloader, threshold, records)


def grid_search_iteration(ce_num, device, filename, from_ground_zero, loss, lr, misleading_binary_masks,
                          misleading_data, misleading_labels, model_confounded, num_of_instances, optim,
                          evaluate_every_nth_epoch, strategy, test_dataloader, threshold, records):
    original_data_size = misleading_data.size(0)

    combination_name = f"ce_num_{ce_num}__lr_{lr}__strategy__{str(strategy)}"
    writer = SummaryWriter(
        log_dir=f"runs/caipi_experiment_{optim}_{original_data_size}__num_of_instances_{num_of_instances}{'_ground_zero' if from_ground_zero else ''}/{combination_name}")

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


    print(f"Checking out {ce_num=}, {strategy=}, {lr=}, {num_of_instances=}")

    grid_model = CNNTwoConv(num_classes, device)
    grid_model.load_state_dict(model_confounded.state_dict())

    if optim == "adam":
        optimizer = Adam(grid_model.parameters(), lr=lr)
    elif optim == "sgd":
        optimizer = SGD(grid_model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optim}")

    metrics_dict, avg_loss = XILUtils.test_loop(test_dataloader, grid_model, loss, device, metric='all')
    accuracy = metrics_dict["accuracy"]
    cohen_kappa = metrics_dict["kappa"]
    writer.add_scalar("Accuracy/val", accuracy, 0)
    writer.add_scalar("Cohen Kappa/val", cohen_kappa, 0)
    writer.add_scalar("Average Loss/val", avg_loss, 0)
    print(f"Initial accuracy: {100 * accuracy:.2f}%")

    current_data = misleading_data.clone()
    current_labels = misleading_labels.clone()
    current_binary_masks = misleading_binary_masks.clone()

    counterexamples_epoch = 1
    duplicate_informative_instances_iteration = 0

    with progressbar.ProgressBar(widgets=widgets, max_value=1e+4) as bar:
        # Update ProgressBar
        bar.update(accuracy * 1e+4)
        # Step Into Fitting Until Convergence
        while accuracy < threshold:
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

            # # Step 2: Get Model Prediction
            # grid_model.eval()
            # with torch.no_grad():
            #     prediction = grid_model(informative_instances)
            # TODO: remove as it is not used, although step is described by caipi framework,
            #   where prediction is used to present to the explainer.

            # Step 3: Get Informative Targets and Corresponding Binary Masks
            informative_targets = current_labels[indices]
            informative_binary_masks = current_binary_masks[indices]

            # Step 4: Query Explanation
            explanation = XILUtils.create_explanation(informative_instances, informative_binary_masks,
                                                      informative_targets, model=grid_model, device=device,
                                                      target_layers=[grid_model[3]])

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

            # Step 7: Fit to new dataset
            grid_train_dl = DataLoader(TensorDataset(current_data, current_labels), batch_size=batch_size, shuffle=True)
            if from_ground_zero:
                model_state_dict = grid_model.state_dict()
            metrics_dict, avg_loss = fit_until_optimum_or_threshold(grid_model, grid_train_dl, test_dataloader, optimizer, loss,
                                                                    threshold=threshold,
                                                                    evaluate_every_nth_epoch=evaluate_every_nth_epoch)
            if from_ground_zero:
                grid_model.load_state_dict(model_state_dict)

            accuracy = metrics_dict["accuracy"]
            cohen_kappa = metrics_dict["kappa"]

            num_of_artifical_instances = len(current_labels) - original_data_size

            writer.add_scalar("Accuracy/val", accuracy, num_of_artifical_instances)
            writer.add_scalar("Cohen Kappa/val", cohen_kappa, num_of_artifical_instances)
            writer.add_scalar("Average Loss/val", avg_loss, num_of_artifical_instances)

            # Update ProgressBar
            bar.update(accuracy * 1e+4)

            # evaluate accuracy
            records.append({
                "ce_num": ce_num,
                "strategy": str(strategy),
                "lr": lr,
                "num_of_instances_per_epoch": num_of_instances,

                "counterexamples_iteration": counterexamples_epoch,

                "accuracy": float(accuracy),
                "cohen_kappa": float(cohen_kappa),
                "average_loss": float(avg_loss),

                "from_ground_zero": from_ground_zero,
                "num_of_artificial_instances": num_of_artifical_instances,
            })
            df = pd.DataFrame(records)
            save_info_to_csv(filename, df)
            print(f"Epoch {counterexamples_epoch}: Accuracy: {100 * accuracy:.2f}%, Avg. Test Loss: {avg_loss:.4f}")

            if num_of_artifical_instances > original_data_size // 2:
                # TODO: change only after the analysis of the results and if it shows that some model could converge
                break

            # update epoch
            counterexamples_epoch += 1
    df = pd.DataFrame(records)
    save_info_to_csv(filename, df)
    writer.add_hparams(
        {"ce_num": ce_num, "strategy": str(strategy), "lr": lr, "num_of_instances_per_epoch": num_of_instances},
        {"accuracy": accuracy, "cohen_kappa": cohen_kappa,
         "average_loss": avg_loss, "num_of_artificial_instances": num_of_artifical_instances}
    )
    writer.close()
    return records


def merge_config_with_args(args: argparse.Namespace, config_manager: ConfigManager) -> argparse.Namespace:
    """Merge config parameters with command line arguments, giving priority to command line."""
    config_args = config_manager.get_program_args_from_config()

    # Create a new namespace with config defaults
    merged_args = argparse.Namespace()

    # First, set config values
    for key, value in config_args.items():
        setattr(merged_args, key, value)

    # Then, override with command line arguments (which have priority)
    for key, value in vars(args).items():
        if not value is None:
            setattr(merged_args, key, value)

    return merged_args


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
        "--current_path", "-p",
        type=Path,
        default=Path(__file__).parent,
        help="Base directory for loading datasets and models"
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

    args = parser.parse_args()

    # Initialize config manager
    config_manager = ConfigManager(args.config, args.config_case) if args.config else ConfigManager()

    # Merge config with args (args have priority)
    final_args = merge_config_with_args(args, config_manager)

    # Print priority system warnings
    config_manager.print_priority_warnings(final_args)

    current_path = final_args.current_path

    # Load the dataset
    dataset = torch.load(current_path / "08_MNIST_output/misleading_dataset.pth", weights_only=False)
    ds_size = final_args.train_dataset_size
    assert ds_size < len(dataset["inputs"]), f"Dataset size: {ds_size} is set bigger than the actual dataset size."
    misleading_ds_train = RRRDataset(
        dataset["inputs"][:ds_size],
        dataset["targets"][:ds_size],
        dataset["binary_masks"][:ds_size],
    )

    model_confounded = CNNTwoConv(num_classes, device)
    model_confounded.load_state_dict(
        torch.load(current_path / "08_MNIST_output/model_confounded.pth", weights_only=True)
    )
    model_confounded = model_confounded.to(device)

    dataset_test = torch.load(current_path / "08_MNIST_output/test_dataset.pth", weights_only=False)
    ds_test = TensorDataset(dataset_test["inputs"], dataset_test["targets"])
    test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    loss = torch.nn.CrossEntropyLoss()
    threshold = final_args.threshold
    output_file = final_args.output_filename
    optimizer = final_args.optimizer

    grid_search(
        output_file,
        misleading_ds_train,
        model_confounded,
        test_dataloader,
        device,
        loss,
        threshold,
        optimizer,
        evaluate_every_nth_epoch=final_args.evaluate_every_nth_epoch,
        from_ground_zero=final_args.train_from_ground_zero,
        config_manager=config_manager,
        specific_case=final_args.config_case,
    )