"""
RRR Experiment: Right for the Right Reasons on MNIST
Implements Algorithm 4.1 from diploma paper
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

from ...rrr_loss import RRRLoss
from ...utils import XILUtils
from ..cnn import CNNTwoConv


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

        # For loss, lower is better
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def create_oversampled_loader(dataset: TensorDataset, corrected_indices: List[int], batch_size: int,
                              shuffle: bool = True) -> DataLoader:
    """
    Create DataLoader that oversamples corrected instances to appear in every batch.

    Strategy: Give corrected instances higher sampling weight so they appear frequently.
    If we have N total samples and k corrected instances, and want k instances in every batch,
    we need each corrected instance to appear ~(N/k) times more frequently.
    """
    n_samples = len(dataset)
    k = len(corrected_indices)

    # Calculate weights
    weights = torch.ones(n_samples, dtype=torch.float32)

    # Oversample corrected instances
    # Target: each corrected instance appears in most batches
    oversample_factor = max(1.0, n_samples / (k * batch_size))
    weights[corrected_indices] = oversample_factor

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=n_samples,
        replacement=True
    )

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def sample_hyperparameters(config: Dict, n_samples: int) -> List[Dict]:
    """Sample hyperparameters from log-uniform distributions"""
    hp_space = config['phase1']['hyperparameters']
    samples = []

    for _ in range(n_samples):
        sample = {}
        for param_name, param_config in hp_space.items():
            if param_config['scale'] == 'log':
                # Log-uniform sampling: 10^uniform(log10(min), log10(max))
                log_min = np.log10(param_config['min'])
                log_max = np.log10(param_config['max'])
                log_val = np.random.uniform(log_min, log_max)
                sample[param_name] = 10 ** log_val
            else:
                # Linear uniform sampling
                sample[param_name] = np.random.uniform(
                    param_config['min'],
                    param_config['max']
                )
        samples.append(sample)

    return samples


def load_data(config: Dict, script_dir: Path) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Load datasets from .pth files"""
    confounded_train = torch.load(
        script_dir / config['data']['confounded_train'],
        weights_only=False
    )

    # Subset train dataset if specified
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

    Returns:
        indices: List of k sampled indices (in dataset coordinate system)
    """
    # Handle Subset wrapper
    if isinstance(dataset, Subset):
        labels = dataset.dataset.tensors[1][dataset.indices]
        base_indices = dataset.indices
    else:
        labels = dataset.tensors[1]
        base_indices = list(range(len(labels)))

    # Find all eight indices (label = [0, 1])
    eight_mask = (labels[:, 1] == 1)
    eight_positions = torch.where(eight_mask)[0].cpu().numpy()

    # Sample k random eights
    sampled_positions = np.random.choice(eight_positions, size=k, replace=False)

    # Map back to dataset indices
    sampled_indices = [base_indices[pos] for pos in sampled_positions]

    return sampled_indices


def create_masked_dataset(dataset, corrected_indices: List[int], device: str) -> Tuple[TensorDataset, List[int]]:
    """
    Create dataset with binary masks:
    - Use actual masks for corrected_indices
    - Use zero masks for all other instances

    Returns:
        TensorDataset: Dataset with masked data
        List[int]: Corrected indices in the returned dataset's coordinate system
    """
    # Handle Subset wrapper
    if isinstance(dataset, Subset):
        inputs = dataset.dataset.tensors[0][dataset.indices]
        labels = dataset.dataset.tensors[1][dataset.indices]
        masks = dataset.dataset.tensors[2][dataset.indices]

        # Map corrected_indices from original dataset space to subset space
        index_mapping = {orig_idx: subset_idx for subset_idx, orig_idx in enumerate(dataset.indices)}
        corrected_subset_indices = [index_mapping[idx] for idx in corrected_indices if idx in index_mapping]
    else:
        inputs = dataset.tensors[0]
        labels = dataset.tensors[1]
        masks = dataset.tensors[2]
        corrected_subset_indices = corrected_indices

    # Create zero masks for all instances
    masked_dataset_masks = torch.zeros_like(masks)

    # Set actual masks for corrected instances
    masked_dataset_masks[corrected_subset_indices] = masks[corrected_subset_indices]

    return TensorDataset(inputs, labels, masked_dataset_masks), corrected_subset_indices


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Single training epoch"""
    model.train()
    total_loss = 0.0

    for batch_idx, (X, y, A) in enumerate(dataloader):
        X, y, A = X.to(device), y.to(device), A.to(device)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y, A)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, loss_fn, device):
    """Validation step"""
    model.eval()
    total_loss = 0.0

    for X, y, A in dataloader:
        X, y, A = X.to(device), y.to(device), A.to(device)
        pred = model(X)
        loss = loss_fn(pred, y, A)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def test(model, dataloader, device):
    """Test on original (unmodified) dataset - returns accuracy"""
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
        model, optimizer, loss_fn,
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
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)

        # Validation
        val_loss = validate(model, val_loader, loss_fn, device)

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
            best_test_acc = test_acc

        # Early stopping check
        if early_stopping(val_loss):
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    return best_test_acc, test_acc


def phase1_single_run(
        hyperparams: Dict,
        config: Dict,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset,
        test_dataset: TensorDataset,
        device: str,
        run_id: int,
        writer: SummaryWriter = None
) -> float:
    """Single training run for Phase 1"""

    # Set seed for this run
    set_seed(config['experiment']['random_seed_base'] + run_id)

    # Sample k=5 corrected instances
    k = config['phase1']['n_init_corrections']
    corrected_indices = sample_corrected_instances(train_dataset, k, device)

    # Create dataset with appropriate masks
    masked_train_dataset, _ = create_masked_dataset(train_dataset, corrected_indices, device)

    # Create dataloaders
    batch_size = config['phase1']['training']['batch_size']
    train_loader = DataLoader(masked_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = CNNTwoConv(config['model']['num_classes'], device)
    model = model.to(device)

    # Initialize optimizer
    optimizer = Adam(
        model.parameters(),
        lr=hyperparams['learning_rate'],
        betas=config['phase1']['training']['optimizer_params']['betas']
    )

    # Initialize loss
    target_layers = [model[config['model']['target_layer_index']]]
    loss_fn = RRRLoss(
        model=model,
        layers_of_interest=target_layers,
        rightreasons_lambda=hyperparams['lambda_1'],
        weight_regularization_lambda=hyperparams['lambda_2'],
        device=device
    )

    # Train with early stopping
    best_test_acc, final_test_acc = train_with_early_stopping(
        model, optimizer, loss_fn,
        train_loader, val_loader, test_loader,
        config['phase1']['training'],
        device,
        writer=writer,
        global_step_offset=run_id * 1000
    )

    return final_test_acc


def phase1_hyperparameter_search(config: Dict, device: str, output_dir: Path):
    """Phase 1: Hyperparameter Optimization"""
    print("=" * 60)
    print("PHASE 1: HYPERPARAMETER OPTIMIZATION")
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
        print(f"  λ₁={hyperparams['lambda_1']:.6f}, "
              f"λ₂={hyperparams['lambda_2']:.6f}, "
              f"lr={hyperparams['learning_rate']:.6f}")

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
                device, run_id, writer
            )

            run_results.append(test_acc)
            print(f"Test Acc: {test_acc * 100:.2f}%")

            if writer:
                writer.close()

        # Store results
        avg_test_acc = np.mean(run_results)
        std_test_acc = np.std(run_results)

        results.append({
            'config_id': config_id,
            'lambda_1': hyperparams['lambda_1'],
            'lambda_2': hyperparams['lambda_2'],
            'learning_rate': hyperparams['learning_rate'],
            'avg_test_acc': avg_test_acc,
            'std_test_acc': std_test_acc,
            **{f'run_{i}_test_acc': run_results[i] for i in range(n_runs)}
        })

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
    print(f"λ₁ = {best_config['lambda_1']:.6f}")
    print(f"λ₂ = {best_config['lambda_2']:.6f}")
    print(f"lr = {best_config['learning_rate']:.6f}")
    print(f"Avg Test Accuracy: {best_config['avg_test_acc'] * 100:.2f}%")
    print("=" * 60)

    # Save best config
    best_hyperparams = {
        'lambda_1': float(best_config['lambda_1']),
        'lambda_2': float(best_config['lambda_2']),
        'learning_rate': float(best_config['learning_rate']),
        'avg_test_acc': float(best_config['avg_test_acc'])
    }

    with open(output_dir / "best_hyperparameters.yaml", 'w') as f:
        yaml.dump(best_hyperparams, f, default_flow_style=False)

    return best_hyperparams


def phase2_single_run(
        k: int,
        best_hyperparams: Dict,
        config: Dict,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset,
        test_dataset: TensorDataset,
        device: str,
        run_id: int,
        writer: SummaryWriter = None
) -> float:
    """Single training run for Phase 2 with k corrections"""

    # Set seed for this run
    set_seed(config['experiment']['random_seed_base'] + run_id)

    # Sample k corrected instances
    corrected_indices = sample_corrected_instances(train_dataset, k, device)

    # Create dataset with appropriate masks
    masked_train_dataset = create_masked_dataset(train_dataset, corrected_indices, device)

    # Create dataloaders
    batch_size = config['phase2']['training']['batch_size']
    train_loader = DataLoader(masked_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = CNNTwoConv(config['model']['num_classes'], device)
    model = model.to(device)

    # Initialize optimizer with best lr
    optimizer = Adam(
        model.parameters(),
        lr=best_hyperparams['learning_rate'],
        betas=config['phase2']['training']['optimizer_params']['betas']
    )

    # Initialize loss with best hyperparameters
    target_layers = [model[config['model']['target_layer_index']]]
    loss_fn = RRRLoss(
        model=model,
        layers_of_interest=target_layers,
        rightreasons_lambda=best_hyperparams['lambda_1'],
        weight_regularization_lambda=best_hyperparams['lambda_2'],
        device=device
    )

    # Train with early stopping
    best_test_acc, final_test_acc = train_with_early_stopping(
        model, optimizer, loss_fn,
        train_loader, val_loader, test_loader,
        config['phase2']['training'],
        device,
        writer=writer,
        global_step_offset=run_id * 1000
    )

    return final_test_acc


def phase2_sensitivity_analysis(config: Dict, best_hyperparams: Dict, device: str, output_dir: Path):
    """Phase 2: User-Correction Sensitivity Analysis"""
    print("\n" + "=" * 60)
    print("PHASE 2: USER-CORRECTION SENSITIVITY ANALYSIS")
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
                device, run_id, writer
            )

            run_results.append(test_acc)
            print(f"Test Acc: {test_acc * 100:.2f}%")

            if writer:
                writer.close()

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
        print(f"k={row['k']:3d}: {row['avg_test_acc'] * 100:5.2f}% ± {row['std_test_acc'] * 100:4.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RRR Experiment: MNIST")
    parser.add_argument('--config', type=str, default='config_rrr.yaml',
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

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent / config['experiment']['output_dir'] / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Device
    device = XILUtils.define_device()
    print(f"Using device: {device}")

    # Run phases
    if args.phase in ['1', 'both']:
        best_hyperparams = phase1_hyperparameter_search(config, device, output_dir)
    else:
        # Load best hyperparameters from previous run
        best_hp_path = output_dir / "best_hyperparameters.yaml"
        if not best_hp_path.exists():
            raise FileNotFoundError(f"Best hyperparameters not found at {best_hp_path}. Run Phase 1 first.")
        with open(best_hp_path, 'r') as f:
            best_hyperparams = yaml.safe_load(f)

    if args.phase in ['2', 'both']:
        phase2_sensitivity_analysis(config, best_hyperparams, device, output_dir)

    print("\n✓ Experiment complete!")


if __name__ == "__main__":
    main()