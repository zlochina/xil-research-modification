import torch
import torch.nn as nn
from torch.autograd import Function


class GuidedBackpropReLU(Function):
    """
    Implementation of Guided Backpropagation for ReLU activations.
    This modifies the backward pass of ReLU to allow gradients to flow
    only when both the input and the gradient are positive.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor):
        # Store input for backward pass
        ctx.save_for_backward(input)
        # Forward pass is the same as a regular ReLU
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Get stored input
        input, = ctx.saved_tensors

        # Guided backprop: only pass gradients where:
        # 1. Input > 0 (standard ReLU backprop)
        # 2. Gradient > 0 (from deconvnet approach)
        grad_input = grad_output.clone()
        # Zero out gradients where input <= 0 (typical ReLU behavior)
        grad_input[input <= 0] = 0
        # Additionally zero out gradients where grad_output <= 0 (from deconvnet)
        grad_input[grad_output <= 0] = 0

        return grad_input


class _GuidedBackpropReLUModule(nn.Module):
    """
    Module that applies GuidedBackpropReLU function.
    This is needed because Functions can't be directly assigned as module attributes.
    """

    def __init__(self):
        super(_GuidedBackpropReLUModule, self).__init__()

    def forward(self, input):
        return GuidedBackpropReLU.apply(input)


class _GuidedBackpropReLUModel(nn.Module):
    """
    Wrapper model that replaces all ReLU activations with GuidedBackpropReLU
    for guided backpropagation visualization.
    """

    def __init__(self, model):
        super(_GuidedBackpropReLUModel, self).__init__()
        self.model = model
        self.model.eval()  # Set model to evaluation mode

        # Replace all ReLU activations
        self._replace_relu_with_guided_relu()

    def _replace_relu_with_guided_relu(self):
        """
        Recursively replaces all ReLU activations in the model
        with GuidedBackpropReLU for guided backpropagation.
        """
        for name, module in self.model.named_children():
            if isinstance(module, nn.ReLU):
                setattr(self.model, name, _GuidedBackpropReLUModule())
            elif len(module._modules) > 0:  # Check if module has children
                # Process the children but keep the original module
                self._replace_all_layer_type_recursive(module, nn.ReLU, _GuidedBackpropReLUModule())

    @staticmethod
    def _replace_all_layer_type_recursive(model, old_layer_type, new_layer):
        for name, layer in model._modules.items():
            if isinstance(layer, old_layer_type):
                model._modules[name] = new_layer
            _GuidedBackpropReLUModel._replace_all_layer_type_recursive(layer, old_layer_type, new_layer)

    def forward(self, input):
        """
        Forward pass through the model.
        """
        return self.model(input)


class GuidedBackpropagation:
    """
    Class for performing guided backpropagation on a model.
    """

    def __init__(self, model, device):
        import copy
        model_copy = copy.deepcopy(model)
        self.model = _GuidedBackpropReLUModel(model_copy.to(device))
        self.model.eval()

    def generate_gradients(self, input_image, target):
        """
        Generate guided backpropagation gradients for an input image.

        Args:
            input_image: Input image tensor (requires_grad=True)
            target: Target tensor for the output (e.g., one-hot encoded class scores)

        Returns:
            Gradients of the same shape as input_image
        """
        # Forward pass
        input_image.requires_grad = True
        model_output: torch.Tensor = self.model(input_image)

        # Backward pass
        model_output.backward(gradient=target)

        # Return gradients
        return input_image.grad.data.clone()

    def __call__(self, input_image:torch.Tensor, target:torch.Tensor, *args, **kwargs):
        return self.generate_gradients(input_image, target)