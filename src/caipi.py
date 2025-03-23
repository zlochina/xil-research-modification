import torch
from abc import ABC, abstractmethod
# CounterExamples Strategy(RandomStrategy, AlternativeValueStrategy, SubstitutionStrategy)
class Strategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x_tensor, explanation, ce_num, *args, **kwargs):
        raise NotImplementedError("Called to abstract class Strategy")

class RandomStrategy(Strategy):
    # for now only Uniform distribution is used
    def __init__(self, low_limit, high_limit, dtype: torch.dtype):
        self.low = low_limit # inclusive
        self.high = high_limit # exclusive
        self.dtype = dtype
        pass

    def generate_random_tensor(self, shape) -> torch.Tensor:
        shape = tuple(shape)
        if self.dtype == torch.float16 or self.dtype == torch.float32 or self.dtype == torch.float64:
            in_r = torch.rand(*shape, dtype=self.dtype)
            random = in_r * (self.high - self.low) + self.low
        else:
            random = torch.randint(self.low, self.high, shape)
        return random

    def __call__(self, x_tensor: torch.Tensor, explanation: torch.Tensor, ce_num: int, *args, **kwargs):
        assert x_tensor.dtype == self.dtype, f"Given {x_tensor.dtype=} is different from initiliased value {self.dtype}."

        batch_size, feature_size = x_tensor.shape
        out = x_tensor.unsqueeze(1).repeat_interleave(ce_num, dim=1)
        explanation = explanation.unsqueeze(1).repeat_interleave(ce_num, dim=1)
        random = self.generate_random_tensor(out.shape).to(x_tensor.device)
        return torch.where(explanation, random, out)

class AlternativeValueStrategy(Strategy):
    def __init__(self):
        pass

    def __call__(self, x_tensor, explanation, ce_num, *args, **kwargs):
        pass

class SubstitutionStrategy(Strategy):
    def __init__(self):
        pass

    def __call__(self, x_tensor, explanation, ce_num, *args, **kwargs):
        pass

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
    assert set(torch.unique(explanation).tolist()) == {0, 1}, f"explanation is not binary."
    batch_size, feature_size = x_tensor.shape

    # apply counter examples strategy
    counter_examples = strategy(x_tensor, explanation, ce_num) # returns tensor of shape (batch_size, ce_num, features)
    assert list(counter_examples.shape) == [batch_size, ce_num, feature_size]

    return counter_examples