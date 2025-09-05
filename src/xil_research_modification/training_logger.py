import os
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import Dict, Any, Optional


class ExperimentLogger:
    """Manages TensorBoard logging for grid search experiments."""

    def __init__(self, base_log_dir: str = "runs", experiment_name: str = None):
        self.base_log_dir = Path(base_log_dir)
        self.experiment_name = experiment_name or f"caipi_lagrange_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = self.base_log_dir / self.experiment_name
        self.current_writer = None
        self.current_combination_id = None

    def create_combination_writer(self, combination_params: Dict[str, Any], combination_id: int) -> SummaryWriter:
        """Create a unique writer for each hyperparameter combination."""
        # Close previous writer if exists
        if self.current_writer:
            self.current_writer.close()

        # Create descriptive directory name
        combo_name = self._generate_combination_name(combination_params, combination_id)
        combo_dir = self.experiment_dir / combo_name

        # Create writer
        self.current_writer = SummaryWriter(log_dir=str(combo_dir))
        self.current_combination_id = combination_id

        # Log hyperparameters at the start
        self.current_writer.add_text("hyperparameters", self._format_hyperparameters(combination_params))

        return self.current_writer

    def _generate_combination_name(self, params: Dict[str, Any], combo_id: int) -> str:
        """Generate a descriptive name for the combination directory."""
        # Extract key parameters for directory name
        key_params = []

        if 'ce_num' in params:
            key_params.append(f"ce{params['ce_num']}")
        if 'strategy' in params:
            strategy_name = str(params['strategy']).split('(')[0].split('.')[-1]  # Get class name
            key_params.append(f"strat_{strategy_name}")
        if 'lr' in params:
            key_params.append(f"lr{params['lr']}")
        if 'num_of_instances' in params:
            key_params.append(f"inst{params['num_of_instances']}")
        if 'lambda_update_constant' in params:
            key_params.append(f"luc{params['lambda_update_constant']:.2f}")
        if 'lambda_update_interval' in params:
            key_params.append(f"lui{params['lambda_update_interval']}")
        if 'initial_lambda' in params:
            key_params.append(f"il{params['initial_lambda']}")

        combo_name = f"combo_{combo_id:03d}_" + "_".join(key_params)
        return combo_name[:100]  # Limit length for filesystem compatibility

    def _format_hyperparameters(self, params: Dict[str, Any]) -> str:
        """Format hyperparameters for text logging."""
        formatted = []
        for key, value in params.items():
            formatted.append(f"**{key}**: {value}")
        return "\n\n".join(formatted)

    def log_training_metrics(self, epoch: int, train_loss: float, train_accuracy: float,
                             original_loss: float, ce_loss: float):
        """Log training metrics."""
        if not self.current_writer:
            return

        self.current_writer.add_scalar("Loss/train_total", train_loss, epoch)
        self.current_writer.add_scalar("Loss/train_original", original_loss, epoch)
        self.current_writer.add_scalar("Loss/train_counterexamples", ce_loss, epoch)
        self.current_writer.add_scalar("Accuracy/train", train_accuracy, epoch)

    def log_test_metrics(self, epoch: int, test_accuracy: float, test_loss: float,
                         original_loss: float, ce_loss: float):
        """Log test metrics."""
        if not self.current_writer:
            return

        self.current_writer.add_scalar("Loss/test_total", test_loss, epoch)
        self.current_writer.add_scalar("Loss/test_original", original_loss, epoch)
        self.current_writer.add_scalar("Loss/test_counterexamples", ce_loss, epoch)
        self.current_writer.add_scalar("Accuracy/test", test_accuracy, epoch)

    def log_validation_metrics(self, epoch: int, validation_accuracy: float, validation_loss: float):
        """Log validation metrics."""
        if not self.current_writer:
            return

        self.current_writer.add_scalar("Loss/validation_total", validation_loss, epoch)
        self.current_writer.add_scalar("Accuracy/validation", validation_accuracy, epoch)

    def log_lambda_values(self, epoch: int, lambdas: Dict[int, float],
                          lambda_accuracies: torch.Tensor):
        """Log lambda values and their corresponding accuracies."""
        if not self.current_writer:
            return

        for i, (lambda_id, lambda_val) in enumerate(lambdas.items()):
            self.current_writer.add_scalar(f"Lambda/value_{lambda_id}", lambda_val, epoch)
            if i < len(lambda_accuracies):
                self.current_writer.add_scalar(f"Lambda/accuracy_{lambda_id}",
                                               lambda_accuracies[i].item(), epoch)

    def log_counterexample_epoch(self, ce_epoch: int, num_artificial_instances: int,
                                 accuracy_improvement: float = None):
        """Log counterexample generation epoch metrics."""
        if not self.current_writer:
            return

        self.current_writer.add_scalar("CounterExamples/epoch", ce_epoch, ce_epoch)
        self.current_writer.add_scalar("CounterExamples/total_artificial_instances",
                                       num_artificial_instances, ce_epoch)
        if accuracy_improvement is not None:
            self.current_writer.add_scalar("CounterExamples/accuracy_improvement",
                                           accuracy_improvement, ce_epoch)

    def log_final_results(self, final_metrics: Dict[str, Any]):
        """Log final hyperparameters and results."""
        if not self.current_writer:
            return

        # Extract hyperparameters and metrics
        hparams = {}
        metrics = {}

        for key, value in final_metrics.items():
            if key in ['ce_num', 'strategy', 'lr', 'num_of_instances_per_epoch',
                       'lambda_update_constant', 'lambda_update_interval',
                       'initial_lambda', 'threshold']:
                # Convert strategy to string representation
                if key == 'strategy':
                    hparams[key] = str(value).split('(')[0].split('.')[-1]
                else:
                    hparams[key] = value
            else:
                metrics[key] = value

        # Log hyperparameters with metrics
        self.current_writer.add_hparams(hparams, metrics)

    def save_model_checkpoint(self, model_state_dict: Dict[str, torch.Tensor],
                              epoch: int = None, is_final: bool = False):
        """Save model checkpoint in the current combination directory."""
        if not self.current_writer:
            return

        checkpoint_dir = Path(self.current_writer.log_dir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        if is_final:
            checkpoint_path = checkpoint_dir / "model_final.pth"
        else:
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}.pth"

        torch.save(model_state_dict, checkpoint_path)

        # Log the checkpoint path
        self.current_writer.add_text("model_checkpoint", str(checkpoint_path))

    def close_current_writer(self):
        """Close the current writer."""
        if self.current_writer:
            self.current_writer.close()
            self.current_writer = None
            self.current_combination_id = None

    def get_experiment_summary(self) -> str:
        """Get a summary of all combinations tried."""
        if not self.experiment_dir.exists():
            return "No experiments found."

        combinations = list(self.experiment_dir.iterdir())
        summary = f"Experiment: {self.experiment_name}\n"
        summary += f"Total combinations: {len(combinations)}\n"
        summary += f"Log directory: {self.experiment_dir}\n"
        summary += "\nCombinations:\n"

        for combo_dir in sorted(combinations):
            if combo_dir.is_dir():
                summary += f"  - {combo_dir.name}\n"

        return summary