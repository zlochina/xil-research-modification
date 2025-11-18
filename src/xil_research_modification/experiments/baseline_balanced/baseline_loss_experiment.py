import torch
import torch.nn as nn
import itertools
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from ..cnn import CNNTwoConv
from ...utils import XILUtils
from ..caipi_grid_search import ConfigManager


# ============================================================================
# DATASET CLASSES
# ============================================================================

class UserCorrectedDataset(Dataset):
    """
    Pre-generates N user-corrected instances by zeroing out pixels marked by binary masks.

    CRITICAL NOTE: This dataset only contains class 8 (the confounded class).
    This creates class imbalance in the training data! The batch sampler must account for this.

    Args:
        confounded_dataset: Original dataset with (inputs, labels, binary_masks)
        pool_size: Number of corrected instances to pre-generate
        device: torch device
    """

    def __init__(self, confounded_dataset: torch.utils.data.TensorDataset, pool_size: int, device: str):
        self.device = device

        # Extract only class 8 instances (the confounded class)
        if isinstance(confounded_dataset, Subset):
            dataset = confounded_dataset.dataset
            indices = confounded_dataset.indices

            inputs = dataset.tensors[0][indices]
            labels = dataset.tensors[1][indices]
            masks = dataset.tensors[2][indices]
        elif isinstance(confounded_dataset, torch.utils.data.TensorDataset):
            inputs = confounded_dataset.tensors[0]
            labels = confounded_dataset.tensors[1]
            masks = confounded_dataset.tensors[2]
        else:
            raise RuntimeError(f"Type of `confounded_dataset`={type(confounded_dataset)}, which is not handled...")

        # Filter for class 8: labels are one-hot, so class 8 is [0, 1]
        is_class_eight = (labels[:, 1] == 1)
        eight_inputs = inputs[is_class_eight]
        eight_labels = labels[is_class_eight]
        eight_masks = masks[is_class_eight]

        # Sample pool_size instances (with replacement if needed)
        num_available = len(eight_inputs)
        if pool_size > num_available:
            print(f"WARNING: Requested pool_size={pool_size} but only {num_available} class 8 instances available.")
            print(f"Sampling with replacement.")

        indices = torch.randint(0, num_available, (pool_size,))

        # Generate corrected instances: zero out pixels marked by mask
        self.corrected_inputs = eight_inputs[indices].clone()
        self.corrected_inputs[eight_masks[indices].bool()] = 0.0

        self.labels = eight_labels[indices]
        self.original_inputs = eight_inputs[indices]  # Keep for visualization
        self.masks = eight_masks[indices]

        # # Move to device
        # self.corrected_inputs = self.corrected_inputs.to(device)
        # self.labels = self.labels.to(device)
        # self.original_inputs = self.original_inputs.to(device)
        # self.masks = self.masks.to(device)

    def __len__(self):
        return len(self.corrected_inputs)

    def __getitem__(self, idx):
        return self.corrected_inputs[idx], self.labels[idx]


class MixedBatchSampler(torch.utils.data.Sampler):
    """
    Custom sampler that creates batches mixing confounded and user-corrected samples.

    Each batch contains:
    - (batch_size - user_corrected_per_batch) samples from confounded dataset
    - user_corrected_per_batch samples from user-corrected dataset

    TODO: Future work - implement dynamic ratio schedule (curriculum learning)
    e.g., start with 64-0 split, gradually increase to 60-4 split over epochs.
    This could help stability in early training.
    """

    def __init__(self, confounded_size: int, corrected_size: int,
                 batch_size: int, user_corrected_per_batch: int):
        self.confounded_size = confounded_size
        self.corrected_size = corrected_size
        self.batch_size = batch_size
        self.user_corrected_per_batch = user_corrected_per_batch
        self.confounded_per_batch = batch_size - user_corrected_per_batch

        # Number of batches determined by confounded dataset size
        self.num_batches = confounded_size // self.confounded_per_batch

    def __iter__(self):
        # Shuffle both datasets
        confounded_perm = torch.randperm(self.confounded_size)
        corrected_perm = torch.randperm(self.corrected_size)

        for i in range(self.num_batches):
            # Sample from confounded dataset
            conf_start = i * self.confounded_per_batch
            conf_end = conf_start + self.confounded_per_batch
            conf_indices = confounded_perm[conf_start:conf_end]

            # Sample from corrected dataset (with replacement via modulo)
            corr_start = (i * self.user_corrected_per_batch) % self.corrected_size
            corr_indices = []
            for j in range(self.user_corrected_per_batch):
                idx = (corr_start + j) % self.corrected_size
                corr_indices.append(corrected_perm[idx])
            corr_indices = torch.tensor(corr_indices)

            # Combine: confounded indices, then corrected indices (offset by confounded_size)
            batch_indices = torch.cat([
                conf_indices,
                corr_indices + self.confounded_size  # Offset for corrected dataset
            ])

            yield batch_indices.tolist()

    def __len__(self):
        return self.num_batches


class CombinedDataset(Dataset):
    """Combines confounded and corrected datasets for use with MixedBatchSampler."""

    def __init__(self, confounded_dataset, corrected_dataset):
        self.confounded = confounded_dataset
        self.corrected = corrected_dataset
        self.confounded_size = len(confounded_dataset)

    def __len__(self):
        return len(self.confounded) + len(self.corrected)

    def __getitem__(self, idx):
        if idx < self.confounded_size:
            # From confounded dataset (returns 3 tensors: input, label, mask)
            return self.confounded[idx]
        else:
            # From corrected dataset (returns 2 tensors: input, label)
            corr_idx = idx - self.confounded_size
            inp, label = self.corrected[corr_idx]
            # Add dummy mask (not used, but keeps batch structure consistent)
            dummy_mask = torch.zeros_like(inp)
            return inp, label, dummy_mask


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class BaselineBalancedLoss(nn.Module):
    """
    L = CE(confounded_batch) + λ * CE(user_corrected_batch)

    Treats user-corrected samples with explicit weighting to balance importance.
    The mask allows the loss to compute separate terms for each data source.
    """

    def __init__(self, lambda_weight: float = 1.0):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                is_user_corrected: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size, num_classes] (one-hot)
            is_user_corrected: [batch_size] (bool mask)
        """
        # Convert one-hot targets to class indices
        target_indices = targets.argmax(dim=-1)

        # Compute per-sample loss
        per_sample_loss = self.ce_loss(logits, target_indices)

        # Separate confounded and corrected losses
        confounded_mask = ~is_user_corrected
        confounded_loss = per_sample_loss[confounded_mask].mean() if confounded_mask.any() else 0.0
        corrected_loss = per_sample_loss[is_user_corrected].mean() if is_user_corrected.any() else 0.0

        # Combine with weighting
        total_loss = confounded_loss + self.lambda_weight * corrected_loss

        return total_loss, confounded_loss, corrected_loss


class BaselineImbalancedLoss(nn.Module):
    """
    L = CE(full_batch)

    Standard cross-entropy treating all samples identically.
    Does not distinguish between confounded and user-corrected samples.
    """

    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                is_user_corrected: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size, num_classes] (one-hot)
            is_user_corrected: [batch_size] (bool mask) - not used, but kept for API consistency
        """
        target_indices = targets.argmax(dim=-1)
        total_loss = self.ce_loss(logits, target_indices)

        # Return dummy values for loss breakdown (for consistent logging)
        return total_loss, torch.tensor(0.0), torch.tensor(0.0)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_is_user_corrected_mask(batch_size: int, user_corrected_per_batch: int, device: str) -> torch.Tensor:
    """
    Creates a boolean mask indicating which samples in the batch are user-corrected.
    The last `user_corrected_per_batch` samples are corrected (due to MixedBatchSampler ordering).
    """
    mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    mask[-user_corrected_per_batch:] = True
    return mask


def train_one_epoch(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer, device: str,
                    user_corrected_per_batch: int, global_epoch: int, writer: SummaryWriter) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        avg_loss: Average loss over all batches
        accuracy: Training accuracy
    """
    model.train()
    total_loss = 0.0
    total_conf_loss = 0.0
    total_corr_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels, _) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Create mask for user-corrected samples
        is_user_corrected = create_is_user_corrected_mask(
            len(inputs), user_corrected_per_batch, device
        )

        # Forward pass
        logits = model(inputs)
        loss, conf_loss, corr_loss = loss_fn(logits, labels, is_user_corrected)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_conf_loss += conf_loss.item() if isinstance(conf_loss, torch.Tensor) else conf_loss
        total_corr_loss += corr_loss.item() if isinstance(corr_loss, torch.Tensor) else corr_loss

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        targets = labels.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += len(inputs)

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_conf_loss = total_conf_loss / num_batches
    avg_corr_loss = total_corr_loss / num_batches
    accuracy = correct / total

    # Log to TensorBoard (per epoch)
    writer.add_scalar('Loss/train_epoch', avg_loss, global_epoch)
    writer.add_scalar('Loss/train_confounded', avg_conf_loss, global_epoch)
    writer.add_scalar('Loss/train_corrected', avg_corr_loss, global_epoch)
    writer.add_scalar('Accuracy/train_epoch', accuracy, global_epoch)

    return avg_loss, accuracy


def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module,
             device: str, global_epoch: int, writer: SummaryWriter,
             split_name: str = 'val') -> Tuple[float, float]:
    """
    Evaluate model on validation or test set.

    Handles both dataset types:
    - Standard datasets: (inputs, labels) - 2 items
    - Combined datasets: (inputs, labels, masks) - 3 items

    Args:
        split_name: 'val' or 'test' for TensorBoard logging

    Returns:
        avg_loss: Average loss
        accuracy: Accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            # Handle both 2-item and 3-item batches
            if len(batch) == 2:
                inputs, labels = batch
            elif len(batch) == 3:
                inputs, labels, _ = batch  # Ignore masks for evaluation
            else:
                raise ValueError(f"Expected batch with 2 or 3 items, got {len(batch)}")

            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            logits = model(inputs)

            # Compute loss (no mask needed for validation)
            dummy_mask = torch.zeros(len(inputs), dtype=torch.bool, device=device)
            loss, _, _ = loss_fn(logits, labels, dummy_mask)

            # Accumulate metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            targets = labels.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += len(inputs)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    # Log to TensorBoard
    writer.add_scalar(f'Loss/{split_name}_epoch', avg_loss, global_epoch)
    writer.add_scalar(f'Accuracy/{split_name}_epoch', accuracy, global_epoch)

    return avg_loss, accuracy


def log_example_images(model: nn.Module, confounded_dataset, corrected_dataset,
                       writer: SummaryWriter, device: str, num_examples: int = 8):
    """
    Log example images to TensorBoard:
    - Confounded samples (original with dots)
    - User-corrected samples (dots removed)
    - Model predictions on both
    """
    model.eval()

    # Sample confounded examples (class 8 only)
    if isinstance(confounded_dataset, Subset):
        dataset = confounded_dataset.dataset
        indices = confounded_dataset.indices

        conf_inputs = dataset.tensors[0][indices]
        conf_labels = dataset.tensors[1][indices]
        conf_masks = dataset.tensors[2][indices]
    elif isinstance(confounded_dataset, torch.utils.data.TensorDataset):
        conf_inputs = confounded_dataset.tensors[0]
        conf_labels = confounded_dataset.tensors[1]
        conf_masks = confounded_dataset.tensors[2]
    else:
        raise RuntimeError(f"Type of `confounded_dataset`={type(confounded_dataset)}, which is not handled...")

    is_class_eight = (conf_labels[:, 1] == 1)
    eight_indices = torch.where(is_class_eight)[0][:num_examples]

    conf_samples = conf_inputs[eight_indices].to(device)

    # Sample corrected examples
    corr_samples = corrected_dataset.corrected_inputs[:num_examples].to(device)
    corr_originals = corrected_dataset.original_inputs[:num_examples]

    # Get predictions
    with torch.no_grad():
        conf_preds = model(conf_samples).argmax(dim=-1)
        corr_preds = model(corr_samples).argmax(dim=-1)

    # Create grids
    conf_grid = vutils.make_grid(conf_samples, nrow=4, normalize=True)
    corr_original_grid = vutils.make_grid(corr_originals, nrow=4, normalize=True)
    corr_grid = vutils.make_grid(corr_samples, nrow=4, normalize=True)

    # Log to TensorBoard
    writer.add_image('Examples/confounded_samples', conf_grid)
    writer.add_image('Examples/corrected_original', corr_original_grid)
    writer.add_image('Examples/corrected_samples', corr_grid)

    # Log predictions as text
    conf_pred_str = f"Confounded preds: {conf_preds.cpu().tolist()}"
    corr_pred_str = f"Corrected preds: {corr_preds.cpu().tolist()}"
    writer.add_text('Predictions/confounded', conf_pred_str)
    writer.add_text('Predictions/corrected', corr_pred_str)


def fit_until_early_stopping(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                             loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                             device: str, user_corrected_per_batch: int,
                             patience: int, writer: SummaryWriter,
                             confounded_dataset, corrected_dataset,
                             global_epoch_start: int, num_user_corrected: int) -> Tuple[Dict[str, float], int]:
    """
    Train until early stopping criterion is met.

    Args:
        global_epoch_start: Starting epoch number for TensorBoard logging
        num_user_corrected: Current number of corrected instances per batch (for logging)

    Returns:
        metrics: Dict with final train/val accuracy and loss
        final_global_epoch: Final epoch number (for next iteration)
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    local_epoch = 0
    global_epoch = global_epoch_start

    while True:
        # Log current num_user_corrected configuration (creates step function in TensorBoard)
        writer.add_scalar('Config/num_user_corrected', num_user_corrected, global_epoch)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            user_corrected_per_batch, global_epoch, writer
        )

        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, loss_fn, device, global_epoch, writer, split_name='val'
        )

        print(f"Global Epoch {global_epoch} (Local {local_epoch}, num_corrected={num_user_corrected}): "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Log example images every 10 global epochs
        if global_epoch % 10 == 0:
            log_example_images(model, confounded_dataset, corrected_dataset, writer, device)

        # Early stopping check
        if val_loss < best_val_loss - 1e-2:  # Require significant improvement
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model checkpoint
            torch.save(model.state_dict(), f"{writer.log_dir}/best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at global epoch {global_epoch} "
                      f"(no improvement for {patience} epochs)")
                break

        local_epoch += 1
        global_epoch += 1

    # Load best model
    model.load_state_dict(torch.load(f"{writer.log_dir}/best_model.pth", weights_only=True))

    # Final evaluation
    final_train_loss, final_train_acc = evaluate(
        model, train_loader, loss_fn, device, global_epoch, writer, split_name='train_final'
    )
    final_val_loss, final_val_acc = evaluate(
        model, val_loader, loss_fn, device, global_epoch, writer, split_name='val_final'
    )

    metrics = {
        'train_loss': final_train_loss,
        'train_accuracy': final_train_acc,
        'val_loss': final_val_loss,
        'val_accuracy': final_val_acc,
        'local_epochs_trained': local_epoch,
        'best_val_loss': best_val_loss
    }

    return metrics, global_epoch


# ============================================================================
# EXPERIMENT MANAGEMENT
# ============================================================================

def grid_search_iteration(config: Dict[str, Any], confounded_train_ds, val_loader, test_loader,
                          base_model_state_dict, device: str, num_classes: int,
                          hyperparams: Dict[str, Any], experiment_name: str) -> List[Dict[str, Any]]:
    """
    Single hyperparameter combination experiment with progressive corrected instance increase.

    Returns:
        List of results, one dict per num_user_corrected iteration
    """
    lr = hyperparams['learning_rate']
    loss_type = hyperparams['loss_type']
    lambda_weight = hyperparams['lambda_weight']
    pool_size = hyperparams['user_corrected_pool_size']

    # Create log directory (shared across all num_user_corrected iterations)
    run_name = f"lr{lr}_loss{loss_type}_lambda{lambda_weight}_pool{pool_size}"
    log_dir = Path(config['tensorboard_log_dir']) / run_name
    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"\n{'=' * 80}")
    print(f"Starting: {run_name}")
    print(f"{'=' * 80}")

    # Initialize model from base (only once per hyperparameter combination)
    model = CNNTwoConv(num_classes, device)
    model.load_state_dict(base_model_state_dict)
    model = model.to(device)

    # Create loss function
    if loss_type == 'balanced':
        loss_fn = BaselineBalancedLoss(lambda_weight)
    elif loss_type == 'imbalanced':
        loss_fn = BaselineImbalancedLoss()
    else:
        raise RuntimeError(f"Provided loss function type (\"{loss_type}\") is not recognised ")

    # Create optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create user-corrected dataset (pool) - fixed for this hyperparameter combination
    corrected_ds = UserCorrectedDataset(confounded_train_ds, pool_size, device)

    # Progressive training loop
    results = []
    num_user_corrected = 1  # Start with 1 corrected instance per batch
    global_epoch = 0
    max_user_corrected = config.get('max_user_corrected_per_batch', config['batch_size'] - 1)

    while True:
        print(f"\n{'-' * 80}")
        print(f"Progressive Training: num_user_corrected = {num_user_corrected}")
        print(f"{'-' * 80}")

        # Create combined dataset and mixed batch sampler for current num_user_corrected
        combined_ds = CombinedDataset(confounded_train_ds, corrected_ds)
        ## Important line!!!!!!!
        user_corrected_per_batch = config['user_corrected_max_per_batch']\
            if num_user_corrected > config['user_corrected_max_per_batch'] else num_user_corrected
        #

        batch_sampler = MixedBatchSampler(
            len(confounded_train_ds), len(corrected_ds),
            config['batch_size'], user_corrected_per_batch
        )
        train_loader = DataLoader(combined_ds, batch_sampler=batch_sampler)

        # Train until early stopping with current configuration
        metrics, global_epoch = fit_until_early_stopping(
            model, train_loader, val_loader, loss_fn, optimizer, device,
            num_user_corrected, config['early_stopping_patience'],
            writer, confounded_train_ds, corrected_ds,
            global_epoch, num_user_corrected
        )

        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device,
                                       global_epoch, writer, split_name='test')
        metrics['test_loss'] = test_loss
        metrics['test_accuracy'] = test_acc

        # Record results for this iteration
        iteration_result = {
            'learning_rate': lr,
            'loss_type': loss_type,
            'lambda_weight': lambda_weight,
            'pool_size': pool_size,
            'num_user_corrected': num_user_corrected,
            'global_epochs_total': global_epoch,
            **metrics
        }
        results.append(iteration_result)

        print(f"\nIteration Result: Val Acc={metrics['val_accuracy']:.4f}, "
              f"Test Acc={test_acc:.4f}, Global Epochs={global_epoch}")

        # Check if threshold reached
        if test_acc >= config['accuracy_threshold']:
            print(f"\n{'=' * 80}")
            print(f"Threshold {config['accuracy_threshold']:.2%} reached!")
            print(f"Final test accuracy: {test_acc:.4f}")
            print(f"{'=' * 80}")
            break

        # Check if max corrected instances reached
        if num_user_corrected >= max_user_corrected:
            print(f"\n{'=' * 80}")
            print(f"Max corrected instances per batch ({max_user_corrected}) reached")
            print(f"Final test accuracy: {test_acc:.4f}")
            print(f"{'=' * 80}")
            break

        # Increment and continue
        num_user_corrected += 1

    # Log final hyperparameters summary to TensorBoard
    hparam_dict = {
        'lr': lr,
        'loss_type': loss_type,
        'lambda': lambda_weight,
        'pool_size': pool_size
    }
    metric_dict = {
        'hparam/final_test_accuracy': test_acc,
        'hparam/final_val_accuracy': metrics['val_accuracy'],
        'hparam/num_corrected_at_convergence': num_user_corrected,
        'hparam/total_epochs': global_epoch
    }
    writer.add_hparams(hparam_dict, metric_dict)

    writer.close()

    return results


def main(args):
    config_path: Path = args.config
    experiment_name: str = args.experiment
    """Main experiment loop."""
    # Load config
    config_manager = ConfigManager(config_path, experiment_name)
    config = config_manager.get_program_args_from_config()

    # Define constants
    current_directory = Path(__file__).parent
    num_classes = 2
    device = XILUtils.define_device()
    print(f"Running on {device}...")

    # Load datasets
    print("Loading datasets...")
    datasets_dir = current_directory / "08MNIST"
    train_ds = torch.load(datasets_dir / "confounded_v1" / "train.pth", weights_only=False)
    test_ds_full = torch.load(datasets_dir / "original" / "test.pth", weights_only=False)

    # Optionally limit training set size (for testing)
    if 'train_dataset_size' in config and config['train_dataset_size'] > 0:
        train_size = config['train_dataset_size']
        indices = torch.randperm(len(train_ds))[:train_size]
        train_ds = Subset(train_ds, indices)

    # Split test set into validation and test
    test_length = len(test_ds_full)
    val_length = int(config['validate_ds_size'] * test_length)
    test_length = test_length - val_length
    test_ds, val_ds = random_split(test_ds_full, [test_length, val_length])

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")

    # Create validation and test loaders (standard, no mixed batching)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    # Load base model
    print("Loading base model...")
    base_model_path = current_directory / "model_confounded.pth"
    base_model_state_dict = torch.load(base_model_path, weights_only=True)

    # Define hyperparameter grid
    lr_values = config_manager.get_parameter('learning_rate', experiment_name, [0.01])
    loss_types = config_manager.get_parameter('loss_type', experiment_name, ['balanced', 'imbalanced'])
    lambda_values = config_manager.get_parameter('lambda_weight', experiment_name, [1.0])
    pool_sizes = config_manager.get_parameter('user_corrected_pool_size', experiment_name, [100])

    hyperparameter_grid = {
        'learning_rate': lr_values if isinstance(lr_values, list) else [lr_values],
        'loss_type': loss_types if isinstance(loss_types, list) else [loss_types],
        'lambda_weight': lambda_values if isinstance(lambda_values, list) else [lambda_values],
        'user_corrected_pool_size': pool_sizes if isinstance(pool_sizes, list) else [pool_sizes]
    }

    print(f"\nHyperparameter Grid:")
    for key, values in hyperparameter_grid.items():
        print(f"  {key}: {values}")
    print()

    combinations = list(itertools.product(*hyperparameter_grid.values()))
    print(f"Total combinations to evaluate: {len(combinations)}\n")

    # Run grid search
    all_results = []
    for combo in combinations:
        hyperparams = {
            'learning_rate': combo[0],
            'loss_type': combo[1],
            'lambda_weight': combo[2],
            'user_corrected_pool_size': combo[3]
        }

        # Returns list of results (one per num_user_corrected iteration)
        iteration_results = grid_search_iteration(
            config, train_ds, val_loader, test_loader,
            base_model_state_dict, device, num_classes,
            hyperparams, experiment_name
        )
        all_results.extend(iteration_results)  # Flatten into single list

    # Save results to CSV
    df = pd.DataFrame(all_results)
    output_path = Path(config['output_filename'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nTotal hyperparameter combinations tested: {len(combinations)}")
    print(f"Total iterations (across all combinations): {len(df)}")
    print(f"\nBest test accuracy: {df['test_accuracy'].max():.4f}")
    best_idx = df['test_accuracy'].idxmax()
    print(f"Best hyperparameters:")
    for key in ['learning_rate', 'loss_type', 'lambda_weight', 'pool_size', 'num_user_corrected']:
        print(f"  {key}: {df.loc[best_idx, key]}")
    print(f"Global epochs to reach best: {df.loc[best_idx, 'global_epochs_total']}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline loss experiments for correcting confounded MNIST model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:

  # Run overfit test (sanity check)
  python baseline_loss_experiment.py --config config_baseline.yaml --experiment overfit_test

  # Find optimal learning rate
  python baseline_loss_experiment.py --config config_baseline.yaml --experiment lr_search

  # Search over lambda weight
  python baseline_loss_experiment.py --config config_baseline.yaml --experiment lambda_search

  # Full grid search
  python baseline_loss_experiment.py --config config_baseline.yaml --experiment full_grid_search

  # Override config parameters from command line
  python baseline_loss_experiment.py --config config_baseline.yaml --experiment lr_search --batch_size 32
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=Path,
        required=True,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--experiment', '-e',
        type=str,
        required=True,
        help='Experiment name (must match a key in config file)'
    )

    # Optional overrides
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--learning_rate', type=float, nargs='+', help='Override learning rate(s)')
    parser.add_argument('--early_stopping_patience', type=int, help='Override early stopping patience')

    args = parser.parse_args()

    # TODO: Implement command-line overrides if needed
    # For now, just use config file

    main(args)