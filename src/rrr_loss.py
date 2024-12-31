from typing import List
import torch.nn as nn
import torch
import torch.nn.functional as F

class RRRLoss(nn.Module):
    def __init__(self, model:nn.Module, layers_of_interest:List[nn.Module], criterion: nn.Module=nn.CrossEntropyLoss(),
                 rightreasons_lambda: float=1., weight_regularization_lambda: float=0., device: str="cpu", *args, **kwargs):
        super(RRRLoss, self).__init__(*args, **kwargs)
        self.right_answers_loss = criterion
        self.l2_right_reasons = rightreasons_lambda
        self.l2_weights = weight_regularization_lambda
        # self.gradients = dict() # TODO: remove or decide if either activations or gradients should be present
        self.activations = dict()
        self.model_parameters = model.parameters()
        self.device = device

        if len(layers_of_interest) == 0:
            # Right answers loss should be computed with respect to inputs, instead of convolutional layer
            self.use_inputs = True
            layer_1st = model[0] # TODO: Maybe other non-Sequential models have other accessing method
            layer_1st.register_forward_hook(self.get_inputs("inp_layer_0"))
        else:
            # Registering backward hooks to save gradients
            self.use_inputs = False
            self.layers_of_interest = layers_of_interest
            for i, layer in enumerate(layers_of_interest):
                # layer.register_backward_hook(self.get_gradients(f"grad_layer_{i}")) TODO: remove
                layer.register_forward_hook(self.get_activations(f"act_layer_{i}"))

    def forward(self, predictions, targets, binary_masks):
        # right answers computation
        if type(predictions) == tuple:
            predictions = predictions[0]
        right_answers_loss = self.right_answers_loss(predictions, targets)

        # right reasons computation
        right_reasons_loss = torch.tensor(0, dtype=torch.float, requires_grad=True).to(self.device)
        if self.l2_right_reasons:
            y_log = torch.log(predictions)
            for layer_name, activations in self.activations.items():
                right_reasons_loss += self.calculate_right_reasons_loss_per_layer(activations, binary_masks, log_probs=y_log)
            right_reasons_loss = self.l2_right_reasons * torch.sum(right_reasons_loss / len(self.activations)) # TODO: maybe remove `/ len(...)` as for now Im suggesting taking mean of right reasons for all layers of interest

        # weight regularization computation
        weight_regularization = torch.tensor(0, dtype=torch.float).to(self.device)
        if self.l2_weights: # if non-zero
            for p in self.model_parameters:
                weight_regularization += (p ** 2).sum()
            weight_regularization *= self.l2_weights

        return right_answers_loss + right_reasons_loss + weight_regularization

    def compute_loss(self):
        # TODO: should be the original forward, but instead returns every part of loss **separately**
        pass

    def calculate_right_reasons_loss_per_layer(self, activations, binary_masks, log_probs):
        grad_outputs = torch.ones_like(log_probs, requires_grad=True) # TODO: not really sure about this, as we've done a logarithm operation on the model outputs.
        gradByLayers = torch.autograd.grad(log_probs, activations, grad_outputs=grad_outputs, create_graph=True)[0] # TODO: im not sure why its returning tuple

        # Scale gradient "images" to the pic size
        descaled_bin_mask = self.descale_mask(binary_masks, gradByLayers.shape)
        broadcasted_binary_masks = descaled_bin_mask.expand_as(gradByLayers)

        A_mul_grad = (broadcasted_binary_masks * gradByLayers) ** 2

        return torch.sum(A_mul_grad)

    # Scaling functions
    @staticmethod
    def scale_grads(grads: torch.Tensor, bin_mask_shape: torch.Size):
        target_size = bin_mask_shape[2:] # [pic_size_1, pic_size_2]

        # Use interpolate to resize the spatial dimensions of grads
        # Mode can be 'bilinear', 'nearest', etc., depending on your needs
        scaled_grads = F.interpolate(grads, size=target_size, mode='bilinear', align_corners=False)

        return scaled_grads

    @staticmethod
    def descale_mask(mask: torch.Tensor, grads_shape: torch.Size):
        target_size = grads_shape[2:]  # [size_1, size_2]

        # Use interpolate to resize the spatial dimensions of mask
        # Mode can be 'bilinear', 'nearest', etc., depending on your needs
        descaled_mask = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)

        return descaled_mask

    # Functions to register hooks

    ## Function to register backward hooks
    def get_gradients(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0]
        return hook

    ## Function to register forward hooks
    def get_activations(self, name):
        def hook(module, input, output):
            self.activations[name] = output
        return hook

    ## Function to register forward hooks with taking input
    def get_inputs(self, name):
        def hook(module, input, output):
            assert len(input) == 1 # If it is failed, then my assumption is wrong
            self.activations[name] = input[0] # For some non-known reason it returns tuple with tensor instead
        return hook