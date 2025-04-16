import json
from pathlib import Path
import torch

from src.rrr_dataset import RRRDataset
from src.utils import XILUtils
from src.experiments.cnn import CNNTwoConv
import itertools
from src.caipi import RandomStrategy, SubstitutionStrategy, AlternativeValueStrategy, MarginalizedSubstitutionStrategy, \
    to_counter_examples_2d_pic
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
        "strategy": [random_strategy, substitution_strategy, marginalized_substitution_strategy, alternative_value_strategy],
        "num_of_instances": [1]
    }
    return parameters_grid


def grid_search(filename: Path, misleading_ds_train, model_confounded, test_dataloader, device, loss, threshold,
                num_classes=2, lr=1e-3):
    parameters_grid = define_paramaters(misleading_ds_train.data, misleading_ds_train.labels)
    combinations = list(itertools.product(*parameters_grid.values()))
    label_translation = dict(zero=torch.tensor((1, 0), device=device), eight=torch.tensor((0, 1), device=device))

    def get_informative_instance(targets, num_of_instances):
        # just take some random eight
        eight = label_translation["eight"]
        indices = (targets == eight).all(dim=1)
        # take a random eight
        indices = torch.randint(high=indices.size(0), size=(num_of_instances,))
        return indices

    def save_info_to_json(filename: Path, info: dict):
        # Ensure the parent directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Write the info dictionary to the file as JSON
        with filename.open('w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    misleading_data = misleading_ds_train.data
    misleading_labels = misleading_ds_train.labels
    misleading_binary_masks = misleading_ds_train.binary_masks
    info_dictionary = []
    for ce_num, strategy, num_of_instances in combinations:
        print(f"Checking out {ce_num=}, {strategy=}")
        grid_model = CNNTwoConv(num_classes, device)
        grid_model.load_state_dict(model_confounded.state_dict())
        adam_optimizer = Adam(grid_model.parameters(), lr=lr)
        accuracy, _ = evaluate(grid_model, test_dataloader, loss, verbose=False)
        print(f"Initial accuracy: {100 * accuracy:.2f}%")

        current_run_info = {f"{ce_num=}": {
            f"{strategy}": {}
        }}
        current_data = misleading_data.clone()
        current_labels = misleading_labels.clone()
        current_binary_masks = misleading_binary_masks.clone()
        epoch = 1
        while accuracy < threshold:
            # take some input
            indices = get_informative_instance(current_labels, num_of_instances)
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
                                                         target=label_translation["eight"]).reshape(-1, 1, 28,
                                                                                                    28)  # TODO update to be dynamic
            # populate dataset with new data
            current_data = torch.vstack((current_data, counterexamples))
            current_labels = torch.vstack((current_labels, (informative_targets.repeat_interleave(ce_num, dim=1))))
            # TODO: check if it is in correct shape, in correct order
            current_binary_masks = torch.vstack(
                (current_binary_masks, informative_binary_masks.repeat_interleave(ce_num, dim=1)))
            # fit
            grid_train_dl = DataLoader(TensorDataset(current_data, current_labels), batch_size=batch_size, shuffle=True)
            train(grid_model, grid_train_dl, adam_optimizer, loss, verbose=False)
            # evaluate accuracy
            # every 10 epochs
            if epoch % 10 == 0:
                print(f"{len(current_labels)=}, {len(current_data)=}, {len(current_binary_masks)=}")
                acc, avg_loss = evaluate(grid_model, test_dataloader, loss, verbose=False)
                current_run_info[f"{ce_num=}"][f"{strategy}"][epoch] = {"accuracy": acc, "average_loss": avg_loss}
                info_dictionary.append(current_run_info)
                save_info_to_json(filename, info_dictionary)
                print(f"Epoch {epoch}: Accuracy: {100 * acc:.2f}%, Avg. Test Loss: {avg_loss:.4f}")
                accuracy = acc
            # update epoch
            epoch += 1


if __name__ == "__main__":
    # Load the dataset
    current_path = Path(__file__).parent
    dataset = torch.load(current_path / "08_MNIST_output/misleading_dataset.pth")
    misleading_ds_train = RRRDataset(dataset["inputs"], dataset["targets"], dataset["binary_masks"])

    model_confounded = CNNTwoConv(num_classes, device)
    model_confounded.load_state_dict(torch.load(current_path / "08_MNIST_output/model_confounded.pth"))
    model_confounded = model_confounded.to(device)

    dataset_test = torch.load(current_path / "08_MNIST_output/test_dataset.pth")
    ds_test = TensorDataset(dataset_test["inputs"], dataset_test["targets"])
    test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    loss = torch.nn.CrossEntropyLoss()
    threshold = 0.95
    grid_search(Path("caipi_expr/caipi_grid_search_1run.json"), misleading_ds_train, model_confounded,
                test_dataloader, device, loss, threshold)