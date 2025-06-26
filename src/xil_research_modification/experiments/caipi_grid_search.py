from pathlib import Path
import torch

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


def define_paramaters(inputs, targets):
    # Initialise strategies

    random_strategy = RandomStrategy(0., 1., torch.float32)
    substitution_strategy = SubstitutionStrategy(inputs, targets)
    marginalized_substitution_strategy = MarginalizedSubstitutionStrategy(inputs, targets)
    alternative_value_strategy = AlternativeValueStrategy(torch.zeros(image_shape, device=device), image_shape)
    parameters_grid = {
        "ce_num": [1, 2, 3, 4, 5],
        "strategy": [substitution_strategy, marginalized_substitution_strategy, alternative_value_strategy],
        # "strategy": [substitution_strategy, marginalized_substitution_strategy, alternative_value_strategy, random_strategy],
        "num_of_instances": [5],
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
                evaluate_every_nth_epoch=16, from_ground_zero=False):
    parameters_grid = define_paramaters(misleading_ds_train.data.to(device), misleading_ds_train.labels.to(device))
    combinations = list(itertools.product(*parameters_grid.values()))

    misleading_data = misleading_ds_train.data.to(device)
    misleading_labels = misleading_ds_train.labels.to(device)
    misleading_binary_masks = misleading_ds_train.binary_masks.to(device)



    for ce_num, strategy, num_of_instances, lr in combinations:
        grid_search_iteration(ce_num, device, filename, from_ground_zero, loss, lr, misleading_binary_masks,
                              misleading_data, misleading_labels, model_confounded, num_of_instances, optim,
                              evaluate_every_nth_epoch, strategy, test_dataloader, threshold)


def grid_search_iteration(ce_num, device, filename, from_ground_zero, loss, lr, misleading_binary_masks,
                          misleading_data, misleading_labels, model_confounded, num_of_instances, optim,
                          evaluate_every_nth_epoch, strategy, test_dataloader, threshold):

    combination_name = f"ce_num_{ce_num}__lr_{lr}__num_of_instances_{num_of_instances}__strategy__{str(strategy)}{'ground_zero' if from_ground_zero else ''}"
    writer = SummaryWriter(log_dir=f"runs/caipi_experiment/{combination_name}")

    records = []
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

    original_data_size = misleading_data.size(0)

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
    kappa = metrics_dict["kappa"]
    writer.add_scalar("Accuracy/val", accuracy, 1)
    writer.add_scalar("Cohen Kappa/val", kappa, 1)
    writer.add_scalar("Average Loss/val", avg_loss, 1)
    print(f"Initial accuracy: {100 * accuracy:.2f}%")

    current_data = misleading_data.clone()
    current_labels = misleading_labels.clone()
    current_binary_masks = misleading_binary_masks.clone()

    counterexamples_epoch = 1

    with progressbar.ProgressBar(widgets=widgets, max_value=1e+4) as bar:
        # Update ProgressBar
        bar.update(accuracy * 1e+4)
        # Step Into Fitting Until Convergence
        while accuracy < threshold:
            # Step 1: Get Informative Instances
            indices = get_informative_instance(current_labels[:original_data_size], num_of_instances,
                                               original_data_size)
            informative_instances = current_data[indices]

            # Step 2: Get Model Prediction
            grid_model.eval()
            with torch.no_grad():
                prediction = grid_model(informative_instances)

            # TODO: special case. I've got no idea what to do, when prediction is wrong
            # get indices of NOT the special case (exclusion)
            special_case_indices = torch.where(prediction.argmax(dim=1) == current_labels[indices].argmax(dim=1))[0]
            if len(special_case_indices) != len(indices):
                # update prediction, informative_instances and indices
                informative_instances = informative_instances[special_case_indices]
                indices = indices[special_case_indices]
                prediction = prediction[special_case_indices]

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

            writer.add_scalar("Accuracy/val", accuracy, counterexamples_epoch)
            writer.add_scalar("Cohen Kappa/val", kappa, counterexamples_epoch)
            writer.add_scalar("Average Loss/val", avg_loss, counterexamples_epoch)

            # Update ProgressBar
            bar.update(accuracy * 1e+4)

            # evaluate accuracy
            print(f"Number of artificial instances {len(current_labels) - original_data_size}.")
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
                "num_of_artificial_instances": len(current_labels) - original_data_size,
            })
            df = pd.DataFrame(records)
            save_info_to_csv(filename, df)
            print(f"Epoch {counterexamples_epoch}: Accuracy: {100 * accuracy:.2f}%, Avg. Test Loss: {avg_loss:.4f}")

            if len(current_labels) - original_data_size > 500:
                break

            # update epoch
            counterexamples_epoch += 1
    df = pd.DataFrame(records)
    save_info_to_csv(filename, df)
    writer.add_hparams(
        {"ce_num": ce_num, "strategy": str(strategy), "lr": lr, "num_of_instances_per_epoch": num_of_instances},
        {"accuracy": accuracy, "cohen_kappa": cohen_kappa, "average_loss": avg_loss}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run grid search with counterexample injection and save results to CSV"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.95,
        help="Accuracy threshold at which to stop each run"
    )
    parser.add_argument(
        "--output_filename", "-o",
        type=Path,
        default=Path("caipi_expr/caipi_grid_search_1run.csv"),
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
        default="sgd",
        choices=["adam", "sgd"],
        help="Optimizer to use for training the model"
    )
    parser.add_argument(
        "--train_from_ground_zero",
        action="store_true",
        help="If set, train from ground zero instead of using pretrained model with counterexmples from previous epochs",
        default=False
    )
    parser.add_argument(
        "--evaluate_every_nth_epoch",
        type=int,
        default=10,
        help="Number of epochs without improvement before stopping training"
    )
    parser.add_argument(
        "--train_dataset_size",
        type=int,
        default=-1,
        help="Size of train dataset, Default value set to -1 corresponding to the whole dataset"
    )

    args = parser.parse_args()

    current_path = args.current_path

    # Load the dataset
    dataset = torch.load(current_path / "08_MNIST_output/misleading_dataset.pth")
    ds_size = args.train_dataset_size
    assert ds_size < len(dataset["inputs"]), f"Dataset size: {ds_size} is set bigger than the actual dataset size."
    misleading_ds_train = RRRDataset(
        dataset["inputs"][:ds_size],
        dataset["targets"][:ds_size],
        dataset["binary_masks"][:ds_size],
    )

    model_confounded = CNNTwoConv(num_classes, device)
    model_confounded.load_state_dict(
        torch.load(current_path / "08_MNIST_output/model_confounded.pth")
    )
    model_confounded = model_confounded.to(device)

    dataset_test = torch.load(current_path / "08_MNIST_output/test_dataset.pth")
    ds_test = TensorDataset(dataset_test["inputs"], dataset_test["targets"])
    test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    loss = torch.nn.CrossEntropyLoss()
    threshold = args.threshold
    output_file = args.output_filename
    optimizer = args.optimizer

    grid_search(
        output_file,
        misleading_ds_train,
        model_confounded,
        test_dataloader,
        device,
        loss,
        threshold,
        optimizer,
        evaluate_every_nth_epoch=args.evaluate_every_nth_epoch,
        from_ground_zero=args.train_from_ground_zero,
    )