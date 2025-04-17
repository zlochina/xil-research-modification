import torch
from abc import ABC, abstractmethod

# Implementation of CAIPI framework from
# https://dl.acm.org/doi/pdf/10.1145/3306618.3314293

# CounterExamples Strategy(RandomStrategy, AlternativeValueStrategy, SubstitutionStrategy)
class Strategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x_tensor: torch.Tensor, explanation: torch.Tensor, ce_num: int, *args, **kwargs):
        raise NotImplementedError("Called to abstract class Strategy")

    def __str__(self):
        return self.__class__.__name__

class RandomStrategy(Strategy):
    # for now only Uniform distribution is used
    def __init__(self, low_limit, high_limit, dtype: torch.dtype):
        self.low = low_limit # inclusive
        self.high = high_limit # exclusive
        self.dtype = dtype
        pass

    def _generate_random_tensor(self, shape) -> torch.Tensor:
        shape = tuple(shape)
        if self.dtype == torch.float16 or self.dtype == torch.float32 or self.dtype == torch.float64:
            in_r = torch.rand(*shape, dtype=self.dtype)
            random = in_r * (self.high - self.low) + self.low
        else:
            random = torch.randint(self.low, self.high, shape)
        return random

    def __call__(self, x_tensor: torch.Tensor, explanation: torch.Tensor, ce_num: int, *args, **kwargs):
        assert x_tensor.dtype == self.dtype, f"Given {x_tensor.dtype=} is different from initiliased value {self.dtype}."

        out = x_tensor.unsqueeze(1).repeat_interleave(ce_num, dim=1)
        explanation = explanation.unsqueeze(1).repeat_interleave(ce_num, dim=1)
        random = self._generate_random_tensor(out.shape).to(x_tensor.device)
        return torch.where(explanation, random, out)


class AlternativeValueStrategy(Strategy):
    """Strategy to replace selected features with alternative values.

    The strategy supports several forms of alternative values:
    1. Single value: A single tensor to use for all replacements
    2. Multiple choice values: A collection of N alternative values to sample from
    3. Full tensor: A tensor matching the input shape to use for replacements
    4. Multiple full tensors: N different complete alternatives to sample from
    """

    def __init__(self, alternative_value: torch.Tensor, x_tensor_shape: torch.Size):
        """Initialize with alternative values to use for replacements.

        Args:
            alternative_value: A tensor containing alternative values
            x_tensor_shape: Expected shape of input tensors to create counter examples for
        """
        assert isinstance(alternative_value, torch.Tensor), "Alternative value must be a torch.Tensor"

        self.alt_value = alternative_value
        self.alt_value_shape = alternative_value.shape
        self.x_tensor_shape = x_tensor_shape

        # Check dimensionality relationship between alternative values and input tensor
        alt_value_dimensions = len(self.alt_value_shape)
        x_tensor_dimensions = len(self.x_tensor_shape)

        # Check if alternative value has correct dimensions
        assert alt_value_dimensions == x_tensor_dimensions or alt_value_dimensions == x_tensor_dimensions + 1, \
            f"Alternative value shape dimensions {alt_value_dimensions} must match input tensor dimensions {x_tensor_dimensions} or {x_tensor_dimensions + 1}"

        # Determine if we have multiple choices (N different alternatives)
        self._multiple_choices = alt_value_dimensions == x_tensor_dimensions + 1

        if self._multiple_choices:
            self._N = self.alt_value_shape[0]  # Number of alternatives to choose from

    def __call__(self, x_tensor: torch.Tensor, explanation: torch.Tensor, ce_num: int, *args, **kwargs):
        """Generate counter examples by replacing features according to explanation mask.

        Args:
            x_tensor: Input tensor of shape [batch_size, features]
            explanation: Binary mask of shape [batch_size, features]
            ce_num: Number of counter examples to generate per input

        Returns:
            Counter examples tensor of shape [batch_size, ce_num, features]
        """
        # Verify input shape matches expected shape
        assert tuple(x_tensor.shape[1:]) == tuple(self.x_tensor_shape[1:]), \
            f"Input tensor feature dimensions {tuple(x_tensor.shape[1:])} don't match expected {tuple(self.x_tensor_shape[1:])}"

        batch_size = x_tensor.shape[0]

        # Create output tensor by repeating input
        out = x_tensor.unsqueeze(1).repeat_interleave(ce_num, dim=1)
        explanation = explanation.unsqueeze(1).repeat_interleave(ce_num, dim=1)

        # Handle different cases based on multiple_choices and shape
        if self._multiple_choices:
            # Case: We have N different alternatives to sample from

            # Sample with replacement if needed
            replacement_needed = ce_num >= self._N

            # Generate random indices for sampling
            if len(self.alt_value_shape) == 2 and self.alt_value_shape[1] == 1:
                # Case: Multiple scalar values [N, 1]
                # Sample indices for each position
                indices = torch.randint(
                    0, self._N,
                    (batch_size, ce_num),
                    device=x_tensor.device
                )

                # Select values and expand to match output shape
                alt_selected = self.alt_value[indices].expand_as(out)

            else:
                # Case: Multiple full tensors [N, ...]
                # Sample one alternative per counter example
                indices = torch.randint(
                    0, self._N,
                    (batch_size, ce_num),
                    device=x_tensor.device
                )

                # Select alternatives based on sampled indices
                # Shape: [batch_size, ce_num, ...feature_dims]
                alt_selected = self.alt_value[indices]
        else:
            # Case: Single alternative (either scalar or full tensor)
            if self.alt_value_shape == torch.Size([1]):
                # Single scalar value
                alt_selected = self.alt_value.expand_as(out)
            else:
                # Full tensor matching input shape
                alt_selected = self.alt_value.unsqueeze(1).repeat_interleave(ce_num, dim=1)

        # Apply replacements according to explanation mask
        return torch.where(explanation, alt_selected, out)

class SubstitutionStrategy(Strategy):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self._inputs = inputs
        self._targets = targets
        pass

    def __call__(self, x_tensor: torch.Tensor, explanation: torch.Tensor, ce_num: int, *args, **kwargs):
        target: torch.Tensor | None = kwargs.get("target", None)
        if target is None:
            raise ValueError("Target is not provided.")
        # Step 1: Match target to self._targets
        # Find indices where target matches self._targets
        target_indices = torch.where((self._targets == target).all(dim=1))[1]

        # Select corresponding inputs for the matched targets
        matched_inputs = self._inputs[target_indices]

        # Broadcast to create ce_num copies
        out = x_tensor.unsqueeze(1).repeat_interleave(ce_num, dim=1)
        explanation = explanation.unsqueeze(1).repeat_interleave(ce_num, dim=1)

        # Step 2: Randomly choose input for each counter example
        # Generate random indices for selecting inputs
        random_indices = torch.randint(0, matched_inputs.shape[0], (out.shape[0], ce_num),
                                       device=out.device, dtype=torch.long)

        # Select randomly chosen inputs
        random_inputs = matched_inputs[random_indices, torch.arange(out.shape[1]).unsqueeze(0)]

        # Step 3: Replace elements marked by binary mask explanation with randomly chosen inputs
        return torch.where(explanation, random_inputs, out)

class MarginalizedSubstitutionStrategy(Strategy):
    # TODO: An idea, which will not be implemented for 08 MNIST experiment,
    # but could be used for other experiments is:
    # expecting to receive an additional argument, which will specify, to which targets, we should
    # expand. For now implementation would be marginalisation over all targets.

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self._inputs = inputs
        self._targets = targets
        pass

    def __call__(self, x_tensor: torch.Tensor, explanation: torch.Tensor, ce_num: int, *args, **kwargs):
        # Step 1: Match target to self._targets. REDUNDANT
        matched_inputs = self._inputs

        # Broadcast to create ce_num copies
        out = x_tensor.unsqueeze(1).repeat_interleave(ce_num, dim=1)
        explanation = explanation.unsqueeze(1).repeat_interleave(ce_num, dim=1)
        
        # Step 2: Randomly choose input for each counter example
        # Generate random indices for selecting inputs
        random_indices = torch.randint(0, matched_inputs.shape[0], (out.shape[0], ce_num),
                                        device=out.device, dtype=torch.long)
        # Select randomly chosen inputs
        random_inputs = matched_inputs[random_indices, torch.arange(out.shape[1]).unsqueeze(0)]

        # Step 3: Replace elements marked by binary mask explanation with randomly chosen inputs
        return torch.where(explanation, random_inputs, out)

# To Counter Examples
# Notes:
#   1. `explanation` shape is dependent on `x_tensor` shape
#   2. Shape of x_tensor: Batch_size X features
#   3. `explanation` should be a binary mask
def to_counter_examples(strategy: Strategy, x_tensor: torch.Tensor, explanation: torch.Tensor, ce_num: int):
    # assert 1. note
    assert explanation.shape == x_tensor.shape,\
        f"Explanation shape {explanation.shape} should be same as x_tensor shape {x_tensor.shape}, but is not."
    # assert 2. note
    assert len(x_tensor.shape) == 2,\
        f"x_tensor shape {x_tensor.shape} should be in form of (batch_size, features_size), but is not."
    # assert 3. note
    assert set(torch.unique(explanation).tolist()) <= {0, 1}, f"explanation is not binary."
    batch_size, feature_size = x_tensor.shape

    # apply counter examples strategy
    counter_examples = strategy(x_tensor, explanation, ce_num) # returns tensor of shape (batch_size, ce_num, features)
    assert list(counter_examples.shape) == [batch_size, ce_num, feature_size]

    return counter_examples

def to_counter_examples_2d_pic(strategy: Strategy, x_tensor: torch.Tensor, explanation: torch.Tensor, ce_num: int, **kwargs):
    # assert 1. note
    assert explanation.shape == x_tensor.shape,\
        f"Explanation shape {explanation.shape} should be same as x_tensor shape {x_tensor.shape}, but is not."
    # assert 2. note
    assert len(x_tensor.shape) == 4,\
        f"x_tensor shape {x_tensor.shape} should be in form of (batch_size, channels, height, width), but is not."
    # assert 3. note
    assert set(torch.unique(explanation).tolist()) <= {0, 1}, f"explanation is not binary."
    batch_size, channels, height, width = x_tensor.shape

    # apply counter examples strategy
    counter_examples = strategy(x_tensor, explanation, ce_num, **kwargs) # returns tensor of shape (batch_size, ce_num, features)
    assert list(counter_examples.shape) == [batch_size, ce_num, channels, height, width], \
        f"Counter examples shape {counter_examples.shape} should be in form of (batch_size, ce_num, channels, height, width), but is not."

    return counter_examples