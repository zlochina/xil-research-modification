from pathlib import Path
import torch

from src.rrr_dataset import RRRDataset
from src.utils import XILUtils
from src.experiments.cnn import CNNTwoConv
import itertools
from src.caipi import RandomStrategy, SubstitutionStrategy, AlternativeValueStrategy, MarginalizedSubstitutionStrategy, \
    to_counter_examples_2d_pic
from torch.optim import Adam, SGD
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import argparse

image_shape = torch.Size((1, 1, 28, 28))
device = XILUtils.define_device()
batch_size = 64
num_classes = 2

def train(model, train_dataloader, optimizer, loss_fn, verbose=True):
    model.train()
    if verbose:
        print("\ntraining...")
    interval = len(train_dataloader) // 10
    for i, (x_batch, y_batch) in enumerate(train_dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred = model(x_batch)
        loss_value = loss_fn(pred, y_batch)
        loss_value.backward()
        optimizer.step()
        if verbose and i * batch_size % interval == 0:
            print(f"Batch {i * batch_size}/{len(train_dataloader.dataset)}: Loss = {loss_value.item():.4f}")


def evaluate(model, test_dataloader, loss_fn, verbose=True):
    if verbose:
        print("\ncomputing score...")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0
        for x_batch, y_batch in test_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            total += y_batch.size(0)
            test_loss += loss_fn(pred, y_batch).item()
            correct += (F.one_hot(pred.argmax(dim=1), num_classes=num_classes) == y_batch).sum().item() / num_classes
            assert correct % 1 == 0

    # Accuracy, average loss
    return correct / total, test_loss / len(test_dataloader)


def train_evaluate(model, train_dataloader, test_dataloader, optimizer, loss_fn, num_epochs=1, verbose=True):
    training_dictionary = {
        "accuracies": [],
        "losses": [],
    }
    for epoch in range(num_epochs):
        if verbose:
            print(f"Epoch {epoch + 1}\n" + 20 * "-")

        # train loop
        train(model, train_dataloader, optimizer, loss_fn, verbose)

        # Evaluate the model
        correct_ratio, avg_test_loss = evaluate(model, test_dataloader, loss_fn, verbose)
        training_dictionary["accuracies"].append(correct_ratio)
        training_dictionary["losses"].append(avg_test_loss)

        if verbose:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Accuracy: {100 * correct_ratio:.2f}%, Avg. Test Loss: {avg_test_loss:.4f}")
    if verbose:
        print("Done training!")
    return training_dictionary


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
        "lr": [1e-3, 1e-4, 1e-5],
    }
    return parameters_grid

def fit_until_optimum_or_threshold(model, train_dataloader, test_dataloader, optimizer, loss_fn, threshold, no_improve_epochs_th=10, evaluate_every_nth_epoch=10):
    # acc_history = []
    epoch = 0
    epochs_no_improve = 0
    best_loss = float('inf')
    eps = 1e-5

    while True:
        train(model, train_dataloader, optimizer, loss_fn, verbose=False)
        if epoch % evaluate_every_nth_epoch == 0:
            acc, avg_loss = evaluate(model, test_dataloader, loss_fn, verbose=False)
            if acc >= threshold:
                print(f"Reached accuracy threshold of {threshold:.2f} at epoch {epoch}.")
                break
            if avg_loss + eps < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= no_improve_epochs_th:
                    print(f"Stopping training after {epochs_no_improve} epochs without improvement.")
                    break
    return acc, avg_loss

def grid_search(filename: Path, misleading_ds_train, model_confounded, test_dataloader, device, loss, threshold, optim,
                num_classes=2, lr=1e-3, save_every_nth_epoch=16, from_ground_zero=False):
    parameters_grid = define_paramaters(misleading_ds_train.data.to(device), misleading_ds_train.labels.to(device))
    combinations = list(itertools.product(*parameters_grid.values()))
    label_translation = dict(zero=torch.tensor((1, 0), device=device), eight=torch.tensor((0, 1), device=device))

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

    def save_info_to_csv(filename: Path, info: pd.DataFrame):
        # Ensure the parent directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Write the info dictionary to the file as CSV
        df.to_csv(filename.with_suffix('.csv'), index=False)

    misleading_data = misleading_ds_train.data.to(device)
    misleading_labels = misleading_ds_train.labels.to(device)
    misleading_binary_masks = misleading_ds_train.binary_masks.to(device)
    records = []

    original_data_size = misleading_data.size(0)

    for ce_num, strategy, num_of_instances, lr in combinations:
        # clear used indices
        used_indices.clear()

        print(f"Checking out {ce_num=}, {strategy=}, {lr=}, {num_of_instances=}")
        grid_model = CNNTwoConv(num_classes, device)
        grid_model.load_state_dict(model_confounded.state_dict())
        optimizer = None
        if optim == "adam":
            optimizer = Adam(grid_model.parameters(), lr=lr)
        elif optim == "sgd":
            optimizer = SGD(grid_model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optim}")
        accuracy, _ = evaluate(grid_model, test_dataloader, loss, verbose=False)
        print(f"Initial accuracy: {100 * accuracy:.2f}%")

        current_data = misleading_data.clone()
        current_labels = misleading_labels.clone()
        current_binary_masks = misleading_binary_masks.clone()
        epoch = 1
        while accuracy < threshold:
            # take some input
            indices = get_informative_instance(current_labels[:original_data_size], num_of_instances, original_data_size)
            informative_instances = current_data[indices]

            # predict input = prediction
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

            # get target, explanation
            informative_targets = current_labels[indices]
            informative_binary_masks = current_binary_masks[indices]
            explanation = XILUtils.create_explanation(informative_instances, informative_binary_masks,
                                                      informative_targets, model=grid_model, device=device,
                                                      target_layers=[grid_model[3]])

            # create counterexamples
            counterexamples = to_counter_examples_2d_pic(strategy, informative_instances, explanation, ce_num,
                                                         target=label_translation["eight"].unsqueeze(0)).reshape(-1, 1, 28,
                                                                                                    28)  # TODO update to be dynamic
            # populate dataset with new data
            current_data = torch.vstack((current_data, counterexamples))
            current_labels = torch.vstack((current_labels, (informative_targets.repeat_interleave(ce_num, dim=0))))
            current_binary_masks = torch.vstack(
                (current_binary_masks, informative_binary_masks.repeat_interleave(ce_num, dim=0)))

            # fit
            grid_train_dl = DataLoader(TensorDataset(current_data, current_labels), batch_size=batch_size, shuffle=True)
            if from_ground_zero:
                model_state_dict = grid_model.state_dict()
            acc, avg_loss = fit_until_optimum_or_threshold(grid_model, grid_train_dl, test_dataloader, optimizer, loss, threshold=threshold,
                                          no_improve_epochs_th=save_every_nth_epoch)
            if from_ground_zero:
                model_confounded.load_state_dict(model_state_dict)
                # # save the model
                # torch.save(model_confounded.state_dict(), current_path / "08_MNIST_output/model_confounded.pth")


            # evaluate accuracy
            print(f"Number of artificial instances {len(current_labels) - original_data_size}.")
            records.append({
                "ce_num": ce_num,
                "strategy": str(strategy),
                "epoch": epoch,
                "accuracy": float(acc),
                "average_loss": float(avg_loss),
                "lr": lr,
            })
            df = pd.DataFrame(records)
            save_info_to_csv(filename, df)
            print(f"Epoch {epoch}: Accuracy: {100 * acc:.2f}%, Avg. Test Loss: {avg_loss:.4f}")
            accuracy = acc
            if epoch * num_of_instances > 2000:
                break

            # update epoch
            epoch += 1
        df = pd.DataFrame(records)
        save_info_to_csv(filename, df)

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
        "--no_improvement_epochs",
        type=int,
        default=10,
        help="Number of epochs without improvement before stopping training"
    )

    args = parser.parse_args()

    current_path = args.current_path

    # Load the dataset
    dataset = torch.load(current_path / "08_MNIST_output/misleading_dataset.pth")
    misleading_ds_train = RRRDataset(
        dataset["inputs"],
        dataset["targets"],
        dataset["binary_masks"],
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
        num_classes=num_classes,
        save_every_nth_epoch=args.no_improvement_epochs,
        from_ground_zero=args.train_from_ground_zero,
    )