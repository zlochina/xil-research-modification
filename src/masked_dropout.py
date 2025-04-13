import torch.nn as nn
import torch

class ScaleLessDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(ScaleLessDropout, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying dropout.
        """
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) >= self.p).float()
        return x * mask

class MaskedDropout(nn.Module):
    def __init__(self, torch_dropout: nn.Module=ScaleLessDropout()):
        super(MaskedDropout, self).__init__()
        self.torch_dropout = torch_dropout

    def forward(self, x: torch.Tensor, explanation_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply masked dropout to the input tensor x using the provided explanation mask.

        Args:
            x (torch.Tensor): Input tensor.
            explanation_mask (torch.Tensor): Explanation mask with the same shape as x.

        Returns:
            torch.Tensor: Output tensor after applying masked dropout.
        """
        # Ensure the explanation mask has the same shape as x
        if x.shape != explanation_mask.shape:
            raise ValueError(
                f"Input tensor and explanation mask must have the same shape. Got {x.shape} and {explanation_mask.shape}."
            )
        # Ensure the explanation mask is a tensor with binary values (0 or 1)
        if not torch.all(torch.logical_or(explanation_mask == 0, explanation_mask == 1)):
            raise ValueError("Explanation mask must contain binary values (0 or 1).")


        # Apply the explanation mask to the input tensor
        x_masked = x * explanation_mask

        # Apply dropout to the masked input tensor
        dropout_result = self.torch_dropout(x_masked)

        # Combine the dropout result with the original input tensor
        return torch.where(explanation_mask.to(dtype=torch.bool), dropout_result, x)