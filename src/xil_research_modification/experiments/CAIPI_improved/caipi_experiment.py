"""
CAIPI Experiment: Counterexample-based XIL on MNIST
Implements Algorithm 4.1 from diploma paper with CAIPI method
"""

from pathlib import Path
import torch
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import argparse
from datetime import datetime
from lovely_tensors import monkey_patch

monkey_patch()

from torch.utils.data import DataLoader, TensorDataset, Subset, WeightedRandomSampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from ...utils import XILUtils
from ..cnn import CNNTwoConv
from ...caipi import (
    RandomStrategy,
    SubstitutionStrategy,
    AlternativeValueStrategy,
    to_counter_examples_2d_pic
)


class ModelSaver():
    def __init__(self):
        self.accuracies = dict()
        self.run = 0

    def iterate(self):
        self.run += 1

    def save_model(self, model):
        model_path = self.output_dir / f'model_{self.id}.pth'
        torch.save(model.state_dict(), model_path)

    def set_id(self, id):
        self.id = id

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def reset(self):
        self.run = 0
        self.accuracies = dict()
        # reset id


# ============================================================================
# Loss Functions (from your baseline implementations)
# ============================================================================

class CAIPIImbalancedLoss(nn.Module):
    """
    Imbalanced Loss: L = CE(full_batch)
    Treats all samples identically (original + counterexamples).
    """

    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_indices = targets.argmax(dim=-1)
        return self.ce_loss(logits, target_indices)


class CAIPIBalancedLoss(nn.Module):
    """
    Balanced Loss: L_original + L_counterexamples
    Equal importance to original and user-corrected data.
    """

    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                is_counterexample: torch.Tensor) -> torch.Tensor:
        """
        Args:
            is_counterexample: [batch_size] bool mask (True for counterexamples)
        """
        target_indices = targets.argmax(dim=-1)
        per_sample_loss = self.ce_loss(logits, target_indices)

        original_mask = ~is_counterexample.to(torch.bool)
        original_loss = per_sample_loss[original_mask].mean() if original_mask.any() else 0.0
        counter_loss = per_sample_loss[is_counterexample].mean() if is_counterexample.any() else 0.0

        return original_loss + counter_loss


class CAIPILagrangeLoss(nn.Module):
    """
    Lagrange Loss: L_original + λ * L_counterexamples
    Explicit weighting for counterexample importance.
    """

    def __init__(self, lambda_weight: float):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                is_counterexample: torch.Tensor) -> torch.Tensor:
        target_indices = targets.argmax(dim=-1)
        per_sample_loss = self.ce_loss(logits, target_indices)

        original_mask = ~is_counterexample
        original_loss = per_sample_loss[original_mask].mean() if original_mask.any() else 0.0
        counter_loss = per_sample_loss[is_counterexample].mean() if is_counterexample.any() else 0.0

        return original_loss + self.lambda_weight * counter_loss


# ============================================================================
# Early Stopping (Reuse from RRR)
# ============================================================================

class EarlyStopping:
    """Early stopping handler"""

    def __init__(self, patience=10, min_delta=0.0, monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    # <PLACEHOLDER: Reuse from RRR script - identical implementation>
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def sample_hyperparameters(config: Dict, n_samples: int) -> List[Dict]:
    """
    Sample hyperparameters for CAIPI.

    Modification strategies are categorical (not continuous).
    """
    hp_space = config['phase1']['hyperparameters']
    samples = []

    # Get strategy names
    strategy_names = hp_space.get('modification_strategy', ['alternative_value'])

    for _ in range(n_samples):
        sample = {}

        # Sample learning rate (log-uniform)
        if 'learning_rate' in hp_space:
            lr_config = hp_space['learning_rate']
            log_min = np.log10(lr_config['min'])
            log_max = np.log10(lr_config['max'])
            sample['learning_rate'] = 10 ** np.random.uniform(log_min, log_max)

        # Sample lambda (log-uniform, only for Lagrange loss)
        if 'lambda_lagrange' in hp_space:
            lambda_config = hp_space['lambda_lagrange']
            log_min = np.log10(lambda_config['min'])
            log_max = np.log10(lambda_config['max'])
            sample['lambda_lagrange'] = 10 ** np.random.uniform(log_min, log_max)

        # Sample modification strategy (categorical)
        sample['modification_strategy'] = np.random.choice(strategy_names)

        samples.append(sample)

    return samples


def load_data(config: Dict, script_dir: Path) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Load datasets from .pth files"""
    # <PLACEHOLDER: Reuse from RRR script - identical implementation>
    confounded_train = torch.load(
        script_dir / config['data']['confounded_train'],
        weights_only=False
    )

    train_size = config['data'].get('train_dataset_size', -1)
    if train_size > 0:
        indices = torch.randperm(len(confounded_train))[:train_size]
        confounded_train = Subset(confounded_train, indices.tolist())

    confounded_test = torch.load(
        script_dir / config['data']['confounded_test'],
        weights_only=False
    )
    original_test = torch.load(
        script_dir / config['data']['original_test'],
        weights_only=False
    )

    return confounded_train, confounded_test, original_test


def sample_corrected_instances(dataset, k: int, device: str) -> List[int]:
    """
    Sample k instances from eights (label=[0,1]) and return their indices
    """
    # <PLACEHOLDER: Reuse from RRR script - identical implementation>
    if isinstance(dataset, Subset):
        labels = dataset.dataset.tensors[1][dataset.indices]
        base_indices = dataset.indices
    else:
        labels = dataset.tensors[1]
        base_indices = list(range(len(labels)))

    eight_mask = (labels[:, 1] == 1)
    eight_positions = torch.where(eight_mask)[0].cpu().numpy()

    sampled_positions = np.random.choice(eight_positions, size=k, replace=False)
    sampled_indices = [base_indices[pos] for pos in sampled_positions]

    return sampled_indices


def get_strategy_instance(strategy_name: str, dataset, device: str):
    """
    Factory function to create strategy instances.

    Args:
        strategy_name: One of ['random', 'substitution', 'alternative_value']
        dataset: Training dataset (for substitution strategy)
        device: Device for tensor operations
    """
    # Extract data and labels from dataset
    if isinstance(dataset, Subset):
        inputs = dataset.dataset.tensors[0][dataset.indices].to(device)
        labels = dataset.dataset.tensors[1][dataset.indices].to(device)
    else:
        inputs = dataset.tensors[0].to(device)
        labels = dataset.tensors[1].to(device)

    image_shape = inputs[0].unsqueeze(0).shape  # (1, C, H, W)

    if strategy_name == 'random':
        return RandomStrategy(0.0, 1.0, torch.float32)
    elif strategy_name == 'substitution':
        return SubstitutionStrategy(inputs, labels)
    elif strategy_name == 'alternative_value':
        return AlternativeValueStrategy(
            torch.zeros(image_shape, device=device),
            image_shape
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def generate_counterexamples(
        corrected_indices: List[int],
        dataset,
        strategy,
        device: str,
        ce_num: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate counterexamples for corrected instances.

    Returns:
        counterexample_data: [k * ce_num, C, H, W]
        counterexample_labels: [k * ce_num, num_classes]
    """
    # Handle Subset wrapper
    if isinstance(dataset, Subset):
        inputs = dataset.dataset.tensors[0][dataset.indices].to(device)
        labels = dataset.dataset.tensors[1][dataset.indices].to(device)
        masks = dataset.dataset.tensors[2][dataset.indices].to(device)

        # Map indices
        index_mapping = {orig_idx: subset_idx for subset_idx, orig_idx in enumerate(dataset.indices)}
        subset_indices = [index_mapping[idx] for idx in corrected_indices]
    else:
        inputs = dataset.tensors[0].to(device)
        labels = dataset.tensors[1].to(device)
        masks = dataset.tensors[2].to(device)
        subset_indices = corrected_indices

    # Get corrected instances
    corrected_inputs = inputs[subset_indices]
    corrected_labels = labels[subset_indices]
    corrected_masks = masks[subset_indices]
    corrected_masks = corrected_masks.to(dtype=torch.bool)

    # Generate counterexamples using your function
    # Binary masks serve as "explanations" marking spurious regions
    label_translation = dict(zero=torch.tensor((1, 0), device=device), eight=torch.tensor((0, 1), device=device))
    counterexamples = to_counter_examples_2d_pic(
        strategy=strategy,
        x_tensor=corrected_inputs,
        explanation=corrected_masks,  # Binary masks as explanations
        ce_num=ce_num,
        target=label_translation["eight"].unsqueeze(0)
    )

    # Reshape: [batch, ce_num, C, H, W] -> [batch * ce_num, C, H, W]
    bs, ce, c, h, w = counterexamples.shape
    counterexample_data = counterexamples.view(bs * ce, c, h, w)

    # Labels: repeat for each counterexample
    counterexample_labels = corrected_labels.repeat_interleave(ce_num, dim=0)

    return counterexample_data.cpu(), counterexample_labels.cpu()


def create_caipi_dataset(
        original_dataset,
        corrected_indices: List[int],
        strategy,
        device: str,
        ce_num: int = 1
) -> Tuple[TensorDataset, torch.Tensor]:
    """
    Create augmented dataset: original + counterexamples.

    Returns:
        TensorDataset with (inputs, labels, is_counterexample_mask)
        is_counterexample_mask: [N_total] bool tensor
    """
    # Extract original data
    if isinstance(original_dataset, Subset):
        original_inputs = original_dataset.dataset.tensors[0][original_dataset.indices]
        original_labels = original_dataset.dataset.tensors[1][original_dataset.indices]
    else:
        original_inputs = original_dataset.tensors[0]
        original_labels = original_dataset.tensors[1]

    n_original = len(original_inputs)

    # Generate counterexamples
    ce_data, ce_labels = generate_counterexamples(
        corrected_indices, original_dataset, strategy, device, ce_num
    )

    # Concatenate
    full_inputs = torch.cat([original_inputs, ce_data], dim=0)
    full_labels = torch.cat([original_labels, ce_labels], dim=0)

    # Create mask: False for original, True for counterexamples
    is_counterexample = torch.zeros(len(full_inputs), dtype=torch.bool)
    is_counterexample[n_original:] = True

    return TensorDataset(full_inputs, full_labels, is_counterexample), is_counterexample


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, loss_fn, device, loss_variant: str):
    """Single training epoch"""
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        if loss_variant == 'imbalanced':
            X, y, _ = batch  # Ignore mask for imbalanced
            X, y = X.to(device), y.to(device)
            loss = loss_fn(model(X), y)
        else:  # balanced or lagrange
            X, y, is_ce = batch
            X, y, is_ce = X.to(device), y.to(device), is_ce.to(device)
            loss = loss_fn(model(X), y, is_ce)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, loss_fn, device, loss_variant: str):
    """Validation step - only uses original confounded data"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            # Validation dataset is just (X, y) - no counterexamples
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)

            if loss_variant == 'imbalanced':
                # Standard CE
                loss = loss_fn(pred, y)
            else:  # balanced or lagrange
                # For validation, treat all samples as "original" (not counterexamples)
                # Create a mask of all False (no counterexamples in validation set)
                is_ce = torch.zeros(len(X), dtype=torch.bool, device=device)
                loss = loss_fn(pred, y, is_ce)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def test(model, dataloader, device):
    """Test on original (unmodified) dataset - returns accuracy"""
    # <PLACEHOLDER: Reuse from RRR script - identical implementation>
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            logits = model(X)
            pred_labels = torch.argmax(logits, dim=-1)
            true_labels = torch.argmax(y, dim=-1)

            all_preds.append(pred_labels)
            all_targets.append(true_labels)

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()

    accuracy = (all_preds == all_targets).mean()
    return accuracy


def train_with_early_stopping(
        model, optimizer, loss_fn, loss_variant: str,
        train_loader, val_loader, test_loader,
        config: Dict, device: str,
        writer: SummaryWriter = None,
        global_step_offset: int = 0
):
    """Train model with early stopping"""
    early_stopping = EarlyStopping(
        patience=config['early_stopping']['patience'],
        min_delta=config['early_stopping']['min_delta'],
        monitor=config['early_stopping']['monitor']
    )

    max_epochs = config['max_epochs']
    best_test_acc = 0.0

    for epoch in range(max_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, loss_variant)

        # Validation
        val_loss = validate(model, val_loader, loss_fn, device, loss_variant)

        # Test (on original data)
        test_acc = test(model, test_loader, device)

        # Logging
        if writer:
            global_step = global_step_offset + epoch
            writer.add_scalar('Loss/train', train_loss, global_step)
            writer.add_scalar('Loss/val', val_loss, global_step)
            writer.add_scalar('Accuracy/test', test_acc, global_step)

        # Track best
        if test_acc > best_test_acc:
            model_saver.save_model(model)
            best_test_acc = test_acc

        # Early stopping check
        if early_stopping(val_loss):
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    return best_test_acc, test_acc


# ============================================================================
# Phase 1: Hyperparameter Search
# ============================================================================

def phase1_single_run(
        hyperparams: Dict,
        config: Dict,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset,
        test_dataset: TensorDataset,
        device: str,
        run_id: int,
        loss_variant: str,
        writer: SummaryWriter = None
) -> float:
    """Single training run for Phase 1"""

    set_seed(config['experiment']['random_seed_base'] + run_id)

    # Sample k=5 corrected instances
    k = config['phase1']['n_init_corrections']
    corrected_indices = sample_corrected_instances(train_dataset, k, device)

    # Get strategy
    strategy = get_strategy_instance(
        hyperparams['modification_strategy'],
        train_dataset,
        device
    )

    # Generate counterexamples and create augmented dataset
    ce_num = config['counterexample']['ce_num']
    caipi_dataset, _ = create_caipi_dataset(
        train_dataset, corrected_indices, strategy, device, ce_num
    )

    # Create dataloaders
    batch_size = config['phase1']['training']['batch_size']
    train_loader = DataLoader(caipi_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model from confounded weights
    model = CNNTwoConv(config['model']['num_classes'], device)
    model = model.to(device)
    script_dir = Path(__file__).parent
    model.load_state_dict(torch.load(
        script_dir / config['data']['model_weights'],
        weights_only=True
    ))

    # Initialize optimizer
    optimizer = Adam(
        model.parameters(),
        lr=hyperparams['learning_rate'],
        betas=config['phase1']['training']['optimizer_params']['betas']
    )

    # Initialize loss function
    if loss_variant == 'imbalanced':
        loss_fn = CAIPIImbalancedLoss()
    elif loss_variant == 'balanced':
        loss_fn = CAIPIBalancedLoss()
    elif loss_variant == 'lagrange':
        loss_fn = CAIPILagrangeLoss(hyperparams['lambda_lagrange'])
    else:
        raise ValueError(f"Unknown loss variant: {loss_variant}")

    model_saver.set_id(f"{loss_variant}_run{run_id}_{hash(str(hyperparams))}")
    model_saver.iterate()
    # Train with early stopping
    best_test_acc, final_test_acc = train_with_early_stopping(
        model, optimizer, loss_fn, loss_variant,
        train_loader, val_loader, test_loader,
        config['phase1']['training'],
        device,
        writer=writer,
        global_step_offset=run_id * 1000
    )

    return final_test_acc


def phase1_hyperparameter_search(config: Dict, device: str, output_dir: Path, loss_variant: str):
    """Phase 1: Hyperparameter Optimization"""
    print("=" * 60)
    print(f"PHASE 1: HYPERPARAMETER OPTIMIZATION ({loss_variant.upper()})")
    print("=" * 60)

    script_dir = Path(__file__).parent

    # Load data
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_data(config, script_dir)

    # Sample hyperparameters
    n_iterations = config['phase1']['n_iterations']
    print(f"Sampling {n_iterations} hyperparameter configurations...")
    hyperparameter_samples = sample_hyperparameters(config, n_iterations)

    # Results storage
    results = []

    # Evaluate each configuration
    n_runs = config['phase1']['n_runs']

    for config_id, hyperparams in enumerate(hyperparameter_samples):
        print(f"\n[Config {config_id + 1}/{n_iterations}]")

        # Print hyperparameters
        print(f"  lr={hyperparams['learning_rate']:.6f}, "
              f"strategy={hyperparams['modification_strategy']}", end="")
        if 'lambda_lagrange' in hyperparams:
            print(f", λ={hyperparams['lambda_lagrange']:.6f}")
        else:
            print()

        run_results = []

        for run_id in range(n_runs):
            print(f"  Run {run_id + 1}/{n_runs}...", end=" ")

            # Create writer for this run
            writer = None
            if config['logging']['tensorboard']:
                log_dir = output_dir / "tensorboard" / f"phase1_config{config_id}_run{run_id}"
                writer = SummaryWriter(log_dir=str(log_dir))

            # Train
            test_acc = phase1_single_run(
                hyperparams, config,
                train_dataset, val_dataset, test_dataset,
                device, run_id, loss_variant, writer
            )

            run_results.append(test_acc)
            print(f"Test Acc: {test_acc * 100:.2f}%")

            if writer:
                writer.close()

        # Store results
        avg_test_acc = np.mean(run_results)
        std_test_acc = np.std(run_results)

        result_dict = {
            'config_id': config_id,
            'learning_rate': hyperparams['learning_rate'],
            'modification_strategy': hyperparams['modification_strategy'],
            'avg_test_acc': avg_test_acc,
            'std_test_acc': std_test_acc,
            **{f'run_{i}_test_acc': run_results[i] for i in range(n_runs)}
        }

        if 'lambda_lagrange' in hyperparams:
            result_dict['lambda_lagrange'] = hyperparams['lambda_lagrange']

        results.append(result_dict)

        print(f"  Average: {avg_test_acc * 100:.2f}% ± {std_test_acc * 100:.2f}%")

    # Save results
    df = pd.DataFrame(results)
    csv_path = output_dir / "phase1_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Phase 1 results saved to {csv_path}")

    # Find best configuration
    best_idx = df['avg_test_acc'].idxmax()
    best_config = df.iloc[best_idx]

    print("\n" + "=" * 60)
    print("BEST HYPERPARAMETERS:")
    print("=" * 60)
    print(f"lr = {best_config['learning_rate']:.6f}")
    print(f"strategy = {best_config['modification_strategy']}")
    if 'lambda_lagrange' in best_config:
        print(f"λ = {best_config['lambda_lagrange']:.6f}")
    print(f"Avg Test Accuracy: {best_config['avg_test_acc'] * 100:.2f}%")
    print("=" * 60)

    # Save best config
    best_hyperparams = {
        'learning_rate': float(best_config['learning_rate']),
        'modification_strategy': str(best_config['modification_strategy']),
        'avg_test_acc': float(best_config['avg_test_acc'])
    }

    if 'lambda_lagrange' in best_config:
        best_hyperparams['lambda_lagrange'] = float(best_config['lambda_lagrange'])

    with open(output_dir / "best_hyperparameters.yaml", 'w') as f:
        yaml.dump(best_hyperparams, f, default_flow_style=False)

    return best_hyperparams


# ============================================================================
# Phase 2: Sensitivity Analysis
# ============================================================================

def phase2_single_run(
        k: int,
        best_hyperparams: Dict,
        config: Dict,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset,
        test_dataset: TensorDataset,
        device: str,
        run_id: int,
        loss_variant: str,
        writer: SummaryWriter = None
) -> float:
    """Single training run for Phase 2 with k corrections"""

    set_seed(config['experiment']['random_seed_base'] + run_id)

    # Sample k corrected instances
    corrected_indices = sample_corrected_instances(train_dataset, k, device)

    # Get strategy
    strategy = get_strategy_instance(
        best_hyperparams['modification_strategy'],
        train_dataset,
        device
    )

    # Generate counterexamples and create augmented dataset
    ce_num = config['counterexample']['ce_num']
    caipi_dataset, _ = create_caipi_dataset(
        train_dataset, corrected_indices, strategy, device, ce_num
    )

    # Create dataloaders
    batch_size = config['phase2']['training']['batch_size']
    train_loader = DataLoader(caipi_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model from confounded weights
    model = CNNTwoConv(config['model']['num_classes'], device)
    model = model.to(device)
    script_dir = Path(__file__).parent
    model.load_state_dict(torch.load(
        script_dir / config['data']['model_weights'],
        weights_only=True
    ))

    # Initialize optimizer with best lr
    optimizer = Adam(
        model.parameters(),
        lr=best_hyperparams['learning_rate'],
        betas=config['phase2']['training']['optimizer_params']['betas']
    )

    # Initialize loss function
    if loss_variant == 'imbalanced':
        loss_fn = CAIPIImbalancedLoss()
    elif loss_variant == 'balanced':
        loss_fn = CAIPIBalancedLoss()
    elif loss_variant == 'lagrange':
        loss_fn = CAIPILagrangeLoss(best_hyperparams['lambda_lagrange'])
    else:
        raise ValueError(f"Unknown loss variant: {loss_variant}")

    # Train with early stopping
    model_saver.set_id(f"{loss_variant}_k{k}_run{run_id}")
    model_saver.iterate()
    best_test_acc, final_test_acc = train_with_early_stopping(
        model, optimizer, loss_fn, loss_variant,
        train_loader, val_loader, test_loader,
        config['phase2']['training'],
        device,
        writer=writer,
        global_step_offset=run_id * 1000
    )

    return final_test_acc


def phase2_sensitivity_analysis(config: Dict, best_hyperparams: Dict, device: str,
                                output_dir: Path, loss_variant: str):
    """Phase 2: User-Correction Sensitivity Analysis"""
    print("\n" + "=" * 60)
    print(f"PHASE 2: USER-CORRECTION SENSITIVITY ANALYSIS ({loss_variant.upper()})")
    print("=" * 60)

    script_dir = Path(__file__).parent

    # Load data
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_data(config, script_dir)

    # Results storage
    results = []

    k_range = config['phase2']['k_range']
    n_runs = config['phase2']['n_runs']

    for k in k_range:
        print(f"\n[k = {k} corrections]")

        run_results = []

        for run_id in range(n_runs):
            print(f"  Run {run_id + 1}/{n_runs}...", end=" ")

            # Create writer for this run
            writer = None
            if config['logging']['tensorboard']:
                log_dir = output_dir / "tensorboard" / f"phase2_k{k}_run{run_id}"
                writer = SummaryWriter(log_dir=str(log_dir))

            # Train
            test_acc = phase2_single_run(
                k, best_hyperparams, config,
                train_dataset, val_dataset, test_dataset,
                device, run_id, loss_variant, writer
            )

            run_results.append(test_acc)
            print(f"Test Acc: {test_acc * 100:.2f}%")

            if writer:
                writer.close()

        model_saver.reset()

        # Store results
        avg_test_acc = np.mean(run_results)
        std_test_acc = np.std(run_results)

        results.append({
            'k': k,
            'avg_test_acc': avg_test_acc,
            'std_test_acc': std_test_acc,
            **{f'run_{i}_test_acc': run_results[i] for i in range(n_runs)}
        })

        print(f"  Average: {avg_test_acc * 100:.2f}% ± {std_test_acc * 100:.2f}%")

    # Save results
    df = pd.DataFrame(results)
    csv_path = output_dir / "phase2_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Phase 2 results saved to {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY:")
    print("=" * 60)
    for _, row in df.iterrows():
        print(f"k={row['k']:3f}: {row['avg_test_acc'] * 100:5.2f}% ± {row['std_test_acc'] * 100:4.2f}%")
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CAIPI Experiment: MNIST")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--phase', type=str, choices=['1', '2', 'both'], default='both',
                        help='Which phase to run')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')

    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Determine loss variant from config filename
    config_name = config_path.stem  # e.g., 'config_caipi_balanced'
    if 'imbalanced' in config_name:
        loss_variant = 'imbalanced'
    elif 'balanced' in config_name:
        loss_variant = 'balanced'
    elif 'lagrange' in config_name:
        loss_variant = 'lagrange'
    else:
        raise ValueError(f"Cannot determine loss variant from config filename: {config_name}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent / config['experiment']['output_dir'] / f"{loss_variant}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    model_saver.set_output_dir(output_dir)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Device
    device = XILUtils.define_device()
    print(f"Using device: {device}")
    print(f"Loss variant: {loss_variant}")

    # Run phases
    if args.phase in ['1', 'both']:
        best_hyperparams = phase1_hyperparameter_search(config, device, output_dir, loss_variant)
    else:
        # Load best hyperparameters from previous run
        best_hp_path = output_dir / "best_hyperparameters.yaml"
        if not best_hp_path.exists():
            raise FileNotFoundError(f"Best hyperparameters not found at {best_hp_path}. Run Phase 1 first.")
        with open(best_hp_path, 'r') as f:
            best_hyperparams = yaml.safe_load(f)

    if args.phase in ['2', 'both']:
        phase2_sensitivity_analysis(config, best_hyperparams, device, output_dir, loss_variant)

    print("\n✓ Experiment complete!")


if __name__ == "__main__":
    model_saver = ModelSaver()
    main()